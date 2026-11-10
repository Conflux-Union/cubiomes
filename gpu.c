/*
 * GPU engine for the 1.18+ Overworld surface biome map and the terrain
 * overview heights, see gpu.h.
 *
 * One process-global engine: dlopen'd OpenCL runtime (opencl_host.c), the
 * kernel sources (gpu_cl_kernels.h), tables packed from the live generator
 * structs, and the dispatches mirroring mapOverworldSurfaceBiome() and
 * mapOverworldTerrainHeight(). Everything
 * that can fail resolves to "unavailable" and the caller falls back to the
 * CPU; a runtime GPU error latches the engine dead for the process.
 */

#include "gpu.h"

#include "gpu_cl_kernels.h"
#include "opencl_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#define GPU_LOCK AcquireSRWLockExclusive(&g_gpu.lock)
#define GPU_UNLOCK ReleaseSRWLockExclusive(&g_gpu.lock)
#else
#include <pthread.h>
#define GPU_LOCK pthread_mutex_lock(&g_gpu.lock)
#define GPU_UNLOCK pthread_mutex_unlock(&g_gpu.lock)
#endif

#include "tables/btree18.h"
#include "tables/btree19.h"
#include "tables/btree192.h"
#include "tables/btree20.h"
#include "tables/btree21wd.h"
#include "tables/btree215.h"
#include "tables/btree262.h"
#include "tables/btree263.h"

// The kernel hardcodes these instance ids and noise indices; pin the enums.
_Static_assert(NP_TEMPERATURE == 0 && NP_HUMIDITY == 1
    && NP_CONTINENTALNESS == 2 && NP_EROSION == 3 && NP_SHIFT == 4
    && NP_WEIRDNESS == 5 && NP_MAX == 6, "kernel climate instance ids");
_Static_assert(TN_SPAGHETTI_3D_RARITY == 0 && TN_SPAGHETTI_3D_THICKNESS == 1
    && TN_SPAGHETTI_3D_1 == 2 && TN_SPAGHETTI_3D_2 == 3
    && TN_SPAGHETTI_ROUGHNESS == 4 && TN_SPAGHETTI_ROUGHNESS_MODULATOR == 5
    && TN_CAVE_ENTRANCE == 6 && TN_NOODLE == 7 && TN_NOODLE_THICKNESS == 8
    && TN_NOODLE_RIDGE_A == 9 && TN_NOODLE_RIDGE_B == 10 && TN_PILLAR == 11
    && TN_PILLAR_RARENESS == 12 && TN_PILLAR_THICKNESS == 13
    && TN_SPAGHETTI_2D == 14 && TN_SPAGHETTI_2D_ELEVATION == 15
    && TN_SPAGHETTI_2D_MODULATOR == 16 && TN_SPAGHETTI_2D_THICKNESS == 17
    && TN_CAVE_LAYER == 18 && TN_CAVE_CHEESE == 19 && TN_MAX == 20,
    "kernel terrain noise indices");
_Static_assert(SP_CONTINENTALNESS == 0 && SP_EROSION == 1
    && SP_RIDGES == 2 && SP_WEIRDNESS == 3, "kernel spline parameter ids");

#if 0 /* parked: surface-columns GPU path - net-wall negative in the product seam per the seeded e2e A/B (opening wall +6.2%, first tile +76%); restore = flip the parked guards in both repos and rebuild */
_Static_assert(AN_BARRIER == 0 && AN_FLOODEDNESS == 1 && AN_SPREAD == 2
    && AN_LAVA == 3 && AN_MAX == 4, "kernel aquifer noise instance ids");
_Static_assert(SURFACE_COLUMN_FLUID == 0x1, "kernel surface column flag");

// Climate + jagged + terrain-noise + aquifer double-perlin instances; the
// aquifer block rides last (kernel ids 27..30) so every offset the older
// kernels' arguments carry stays stable.
#define GPU_INST_COUNT (NP_MAX + 1 + TN_MAX + AN_MAX)
#else
#define GPU_INST_COUNT (NP_MAX + 1 + TN_MAX)
#endif
// Dispatch size cap: ~1M points keeps one output slab bounded (~59 MB worst
// case with the climate-np plane; two rotating slabs ~120 MB); larger
// queries take the CPU fallback and stay correct.
#define GPU_MAX_POINTS (1 << 20)
// Batch-entry block count bound: the per-block failure mask is an int
// bitmask, so n beyond this cannot report failures.
#define GPU_BATCH_MAX 32
#define GPU_CL_OPTS "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt"
#define GPU_LOCAL_SIZE_MAX 128

/*
 * The engine metrics accumulators, one uint64 per gpu.h slot (the layout is
 * API there, mirrored by the mod's Java reader). Written under the engine
 * lock; gpuQueryMetrics() copies them out under the same lock. The event
 * timers stay zero unless the queue was created with profiling enabled.
 */
typedef struct
{
    uint64_t slots[GPU_METRICS_SLOTS];
} GpuMetrics;

typedef struct
{
    int valid;
    int octTotal;
    uint8_t *perm;   // octTotal * 257
    float *pinfo;    // octTotal * 8 (float copy, float kernels)
    double *pinfoD;  // octTotal * 8 (double copy, cfx kernels); slot 7 = h2
    float *pinfoF;   // octTotal * 8 (float mirror of pinfoD, fp32 cfx kernels)
    int noff[GPU_INST_COUNT * 4];
    float namp[GPU_INST_COUNT];
    double nampD[GPU_INST_COUNT];
    float nampF[GPU_INST_COUNT];  // float mirror of nampD, fp32 cfx kernels
    int sNodeTotal;
    int *styp, *slen, *schild;
    float *sloc, *sder;
    int rootSp, rootFactor, rootJagged;
    const uint32_t *btSteps;
    const int32_t *btParam;
    const uint64_t *btNodes;
    size_t btStepsSize, btParamSize, btNodesSize;
    uint32_t btLen, btOrder;
    float snXZScale, snYScale, snXZFactor, snYFactor;
    // The cfx kernels re-run the shim's double chains, so they need the
    // SurfaceNoise scales at full double fidelity (the float copies above
    // serve the float kernels, which were already float-approximate).
    double snXZScaleD, snYScaleD, snXZFactorD, snYFactorD;
    int octMinOff, octMaxOff, octMainOff;
} GpuTables;

/*
 * One output/input slab: the device buffers a dispatch family reads results
 * from (and, for the cfx biome kernel, uploads its caller heights into),
 * all grown together. Two slabs rotate per dispatch under the engine mutex,
 * so while one dispatch waits its readback outside the lock the next one
 * enqueues against the other slab.
 */
typedef struct
{
    cl_mem outNp;  // climate np plane (float biome path)
    cl_mem outId;  // surface biome ids (float + cfx biome paths)
    cl_mem outY;   // float terrain heights
    cl_mem inY;    // cfx biome caller heights
    cl_mem outYi;  // cfx int height output
    cl_mem inX, inZ;   // cfx sparse height coords
    cl_mem inCX, inCZ; // cfx sparse biome cell origins
#if 0 /* parked surface-columns GPU path */
    cl_mem outCol;     // cfx surface columns, 4 ints interleaved per anchor
#endif
    int cap;       // points all nine buffers are sized for
} GpuSlab;

typedef struct
{
#if defined(_WIN32)
    SRWLOCK lock;
#else
    pthread_mutex_t lock;
#endif
    int modeUser;    // -1 when the env/default decides
    int modeEnv;     // resolved env value
    int modeEnvParsed;
    int clInited;
    int clAvail;
    int dead;
    int forceWarned;
    GpuCl cl;
    cl_program cfxProgram;  // fp64 cfx kernels; separate from the float program
    cl_program cfxFloatProgram;  // fp32 cfx kernels (-DCFX_REAL=float build)
    cl_kernel kHeight;  // terrainHeightKernel, created from the built program
    cl_kernel kCfxHeight, kCfxBiome;  // cfxOverviewHeightKernel, cfxBiomeAtYKernel
    cl_kernel kCfxHeightPoints, kCfxBiomePoints;  // their sparse twins
#if 0 /* parked surface-columns GPU path */
    cl_kernel kCfxColPoints;  // cfxSurfaceColumnsPointsKernel
#endif
    cl_kernel kCfxHeightPointsF, kCfxBiomePointsF;  // sparse twins in fp32
#if 0 /* parked surface-columns GPU path */
    cl_kernel kCfxColPointsF;  // the columns twin in fp32
#endif
    int cfxAvail;       // fp64 program built + all four cfx kernels live
    int cfxInited;      // the one lazy cfx build attempt has been made
    int cfxFloatAvail;  // float program built + both sparse float kernels live
    int cfxFloatInited; // the one lazy float cfx build attempt has been made
    size_t localSize;
    GpuTables tables;
    uint64_t keyProbe;
    int keyMc;
    int haveKey;
    int keyFailed;
    uint64_t dispatchCount;
    uint64_t heightCount;
    GpuMetrics metrics;  // gpu.h slot layout
    cl_mem mPerm, mPinfo, mNoff, mNamp;
    cl_mem mPinfoD, mNampD;
    cl_mem mPinfoF, mNampF;  // float mirrors, fp32 cfx kernels
    cl_mem mStyp, mSlen, mSloc, mSder, mSchild;
    cl_mem mSteps, mParam, mNodes;
    GpuSlab slabs[2];
    int activeSlab;  // slab the next dispatch grows/uses, toggled under lock
} GpuEngine;

static GpuEngine g_gpu =
{
#if defined(_WIN32)
    SRWLOCK_INIT,
#else
    PTHREAD_MUTEX_INITIALIZER,
#endif
    -1, GPU_MODE_AUTO,
    // Everything after modeUser/modeEnv is zero-initialized: modeEnvParsed,
    // clInited, clAvail, dead, forceWarned, cl, kHeight, the cfx program and
    // availability flags, localSize (re-set at first init), tables, cache
    // key, counters, slabs.
};

static void gpuClErr(GpuEngine *e, cl_int err, const char *what)
{
    fprintf(stderr, "gpu: %s failed (OpenCL error %d); engine disabled\n",
        what, (int) err);
    e->dead = 1;
}

/* Monotonic host clock for the metrics wall spans (mutex wait, submit,
 * readback wait). Failure leaves zero: metrics only, never a dispatch
 * error. */
static uint64_t nowNanos(void)
{
#if defined(_WIN32)
    LARGE_INTEGER freq, ctr;
    if (!QueryPerformanceFrequency(&freq) || freq.QuadPart <= 0
        || !QueryPerformanceCounter(&ctr))
        return 0;
    uint64_t secs = (uint64_t) ctr.QuadPart / (uint64_t) freq.QuadPart;
    uint64_t rem = (uint64_t) ctr.QuadPart % (uint64_t) freq.QuadPart;
    return secs * 1000000000ULL
        + rem * 1000000000ULL / (uint64_t) freq.QuadPart;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#endif
}

/* Metrics histogram slot of a batch's total point count: ceil(log2(points)),
 * capped at the histogram's top slot (the dispatch cap). Slot 0 is a single
 * point; slot k covers (2^(k-1), 2^k] points. */
static int gpuPointHistSlot(uint64_t points)
{
    int slot = 0;
    while (slot < 20 && (1ULL << slot) < points)
        slot++;
    return slot;
}

static cl_mem gpuBuf(GpuEngine *e, cl_ulong flags, size_t size, void *host)
{
    cl_int err;
    cl_mem m = e->cl.fn.CreateBuffer(e->cl.context, flags, size, host, &err);
    if (!m || err != CL_SUCCESS)
    {
        gpuClErr(e, err, "clCreateBuffer");
        return NULL;
    }
    return m;
}

static void gpuReleasePack(GpuEngine *e)
{
    cl_mem *bufs[] =
    {
        &e->mPerm, &e->mPinfo, &e->mNoff, &e->mNamp,
        &e->mPinfoD, &e->mNampD, &e->mPinfoF, &e->mNampF,
        &e->mStyp, &e->mSlen, &e->mSloc, &e->mSder, &e->mSchild,
        &e->mSteps, &e->mParam, &e->mNodes,
    };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++)
    {
        if (*bufs[i])
            e->cl.fn.ReleaseMemObject(*bufs[i]);
        *bufs[i] = NULL;
    }
    free(e->tables.perm);
    free(e->tables.pinfo);
    free(e->tables.pinfoD);
    free(e->tables.pinfoF);
    free(e->tables.styp);
    free(e->tables.slen);
    free(e->tables.schild);
    free(e->tables.sloc);
    free(e->tables.sder);
    memset(&e->tables, 0, sizeof(e->tables));
    e->haveKey = 0;
    e->keyFailed = 0;
}

static void gpuReleaseSlab(GpuEngine *e, GpuSlab *slab)
{
    if (slab->outNp)
        e->cl.fn.ReleaseMemObject(slab->outNp);
    if (slab->outId)
        e->cl.fn.ReleaseMemObject(slab->outId);
    if (slab->outY)
        e->cl.fn.ReleaseMemObject(slab->outY);
    if (slab->inY)
        e->cl.fn.ReleaseMemObject(slab->inY);
    if (slab->outYi)
        e->cl.fn.ReleaseMemObject(slab->outYi);
    if (slab->inX)
        e->cl.fn.ReleaseMemObject(slab->inX);
    if (slab->inZ)
        e->cl.fn.ReleaseMemObject(slab->inZ);
    if (slab->inCX)
        e->cl.fn.ReleaseMemObject(slab->inCX);
    if (slab->inCZ)
        e->cl.fn.ReleaseMemObject(slab->inCZ);
#if 0 /* parked surface-columns GPU path */
    if (slab->outCol)
        e->cl.fn.ReleaseMemObject(slab->outCol);
#endif
    memset(slab, 0, sizeof(*slab));
}

//----------------------------------------------------------------------
// Mode handling

static int gpuResolveModeLocked(GpuEngine *e)
{
    if (e->modeUser >= 0)
        return e->modeUser;
    if (!e->modeEnvParsed)
    {
        e->modeEnvParsed = 1;
        e->modeEnv = GPU_MODE_AUTO;
        const char *env = getenv("CUBIOMES_GPU");
        if (env)
        {
            if (!strcmp(env, "off"))
                e->modeEnv = GPU_MODE_OFF;
            else if (!strcmp(env, "auto"))
                e->modeEnv = GPU_MODE_AUTO;
            else if (!strcmp(env, "force"))
                e->modeEnv = GPU_MODE_FORCE;
            else
                fprintf(stderr,
                    "gpu: unknown CUBIOMES_GPU=\"%s\", using auto\n", env);
        }
    }
    return e->modeEnv;
}

static void gpuForceWarnLocked(GpuEngine *e, const char *why)
{
    if (e->forceWarned)
        return;
    e->forceWarned = 1;
    fprintf(stderr, "gpu: force mode but %s\n", why);
}

void gpuSetMode(int mode)
{
    if (mode != GPU_MODE_OFF && mode != GPU_MODE_AUTO
        && mode != GPU_MODE_FORCE)
        return;
    GPU_LOCK;
    g_gpu.modeUser = mode;
    GPU_UNLOCK;
}

int gpuGetMode(void)
{
    GPU_LOCK;
    int mode = gpuResolveModeLocked(&g_gpu);
    GPU_UNLOCK;
    return mode;
}

//----------------------------------------------------------------------
// Runtime init

/*
 * The float program is built from the shipped source text byte-for-byte,
 * so devices (and drivers' JIT caches) keep the engine's existing
 * first-use compile behavior untouched. The fp64 cfx kernels compile into
 * a SECOND program, built lazily (see gpuEnsureCfxLocked) and only when
 * the device advertises cl_khr_fp64: a cfx compile failure - or a device
 * without fp64 - then leaves the float path fully intact and only the cfx
 * dispatches report unavailable.
 */
static int gpuDeviceHasFp64(const GpuCl *cl)
{
    size_t extLen = 0;
    if (cl->fn.GetDeviceInfo(cl->device, CL_DEVICE_EXTENSIONS, 0, NULL,
            &extLen) != CL_SUCCESS || extLen == 0)
        return 0;
    char *ext = malloc(extLen);
    if (!ext)
        return 0;
    int has = 0;
    if (cl->fn.GetDeviceInfo(cl->device, CL_DEVICE_EXTENSIONS, extLen, ext,
            NULL) == CL_SUCCESS)
    {
        ext[extLen - 1] = 0;
        has = strstr(ext, "cl_khr_fp64") != NULL;
    }
    free(ext);
    return has;
}

/*
 * Builds one OpenCL program from the given source parts and options. A
 * build failure prints the compiler log once and reports unavailable, so
 * a device that cannot compile a program degrades to the CPU path like
 * any other engine problem.
 */
static cl_program gpuBuildProgram(GpuEngine *e, const char *const *parts,
    const size_t *lens, cl_uint n, const char *opts, const char *what)
{
    cl_int err = CL_SUCCESS;
    cl_program program = e->cl.fn.CreateProgramWithSource(e->cl.context, n,
        (const char **) parts, lens, &err);
    if (!program || err != CL_SUCCESS)
        return NULL;
    err = e->cl.fn.BuildProgram(program, 1, &e->cl.device, opts, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t logLen = 0;
        e->cl.fn.GetProgramBuildInfo(program, e->cl.device,
            CL_PROGRAM_BUILD_LOG, 0, NULL, &logLen);
        if (logLen > 0)
        {
            char *log = (char *) malloc(logLen + 1);
            if (log)
            {
                e->cl.fn.GetProgramBuildInfo(program, e->cl.device,
                    CL_PROGRAM_BUILD_LOG, logLen, log, NULL);
                log[logLen - 1] = 0;
                fprintf(stderr, "gpu: %s program build failed:\n%s\n",
                    what, log);
                free(log);
            }
        }
        e->cl.fn.ReleaseProgram(program);
        return NULL;
    }
    return program;
}

/*
 * The fp64 cfx program: fp64 pragma + the cfx source section, cfx_real
 * left at its double default. Its outputs must stay bit-identical to the
 * shim's CPU chains (the exact-contract gpu-test matrix is the proof).
 */
static cl_program gpuBuildCfxProgram(GpuEngine *e)
{
    const char *parts[2] = { GPU_CL_PRAGMA_FP64, GPU_CL_SRC_CFX };
    const size_t lens[2] =
    {
        strlen(GPU_CL_PRAGMA_FP64),
        strlen(GPU_CL_SRC_CFX),
    };
    return gpuBuildProgram(e, parts, lens, 2, GPU_CL_OPTS, "cfx fp64");
}

/*
 * The fp32 cfx program: the SAME cfx source section with cfx_real=float.
 * -cl-single-precision-constant makes the unsuffixed literals follow the
 * typedef, so the kernels execute the identical expression in single
 * precision - a drift-budgeted approximation, never bit-exact. No fp64
 * pragma or device check: float execution needs no extensions.
 */
static cl_program gpuBuildCfxFloatProgram(GpuEngine *e)
{
    const char *src = GPU_CL_SRC_CFX;
    const size_t len = sizeof(GPU_CL_SRC_CFX) - 1;
    return gpuBuildProgram(e, &src, &len, 1,
        GPU_CL_OPTS " -cl-single-precision-constant -DCFX_REAL=float",
        "cfx float");
}

static cl_kernel gpuKernelFromProgram(GpuEngine *e, cl_program program,
    const char *name)
{
    cl_int err = CL_SUCCESS;
    cl_kernel k = e->cl.fn.CreateKernel(program, name, &err);
    if (!k || err != CL_SUCCESS)
        return NULL;
    return k;
}

static void gpuEnsureRuntimeLocked(GpuEngine *e)
{
    if (e->clInited)
        return;
    e->clInited = 1;
    e->localSize = 64;
    if (gpuClInit(&e->cl) != 0)
        return;
    e->clAvail = 1;
    if (gpuClBuild(&e->cl, GPU_CL_SRC, GPU_CL_OPTS, "surfaceBiomeKernel"))
    {
        e->clAvail = 0;
        return;
    }
    cl_int err = CL_SUCCESS;
    e->kHeight = e->cl.fn.CreateKernel(e->cl.program, "terrainHeightKernel",
        &err);
    if (!e->kHeight || err != CL_SUCCESS)
    {
        e->clAvail = 0;
        return;
    }

    // One local size serves every built kernel: the smallest workgroup
    // limit wins. The cfx program may not exist yet (built lazily on the
    // first cfx dispatch, so float-only consumers never pay its JIT cost);
    // gpuEnsureCfxLocked folds its kernels into the minimum when it does.
    size_t wg = 0;
    e->cl.fn.GetKernelWorkGroupInfo(e->cl.kernel, e->cl.device,
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL);
    size_t wgh = 0;
    e->cl.fn.GetKernelWorkGroupInfo(e->kHeight, e->cl.device,
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgh), &wgh, NULL);
    if (wgh > 0 && (wg == 0 || wgh < wg))
        wg = wgh;
    if (wg > 0)
        e->localSize = wg < GPU_LOCAL_SIZE_MAX ? wg : GPU_LOCAL_SIZE_MAX;
}

/*
 * Lazy fp64 build for the cfx product kernels: compiled on the first cfx
 * dispatch rather than at engine init, so consumers that only ever use the
 * float paths never pay this program's JIT compile cost. Safe under the
 * engine mutex (both cfx dispatches hold it); cfxInited latches the single
 * attempt - a failed build (no fp64, compiler failure) is not retried, and
 * the dispatches fall back to the CPU for the process lifetime like any
 * other engine problem.
 */
static int gpuEnsureCfxLocked(GpuEngine *e)
{
    if (e->cfxInited)
        return e->cfxAvail;
    e->cfxInited = 1;
    /* Without a live runtime (no loader, no GPU) the fn table stays all
     * NULL and gpuDeviceHasFp64 would call through a zeroed function
     * pointer; refuse like gpuEnsureCfxFloatLocked. */
    if (!e->clAvail)
        return 0;
    if (!gpuDeviceHasFp64(&e->cl))
        return 0;
    e->cfxProgram = gpuBuildCfxProgram(e);
    if (!e->cfxProgram)
        return 0;
    e->kCfxHeight = gpuKernelFromProgram(e, e->cfxProgram,
        "cfxOverviewHeightKernel");
    e->kCfxBiome = gpuKernelFromProgram(e, e->cfxProgram,
        "cfxBiomeAtYKernel");
    e->kCfxHeightPoints = gpuKernelFromProgram(e, e->cfxProgram,
        "cfxOverviewHeightPointsKernel");
    e->kCfxBiomePoints = gpuKernelFromProgram(e, e->cfxProgram,
        "cfxBiomeAtYPointsKernel");
#if 0 /* parked surface-columns GPU path */
    e->kCfxColPoints = gpuKernelFromProgram(e, e->cfxProgram,
        "cfxSurfaceColumnsPointsKernel");
#endif
    e->cfxAvail = e->kCfxHeight && e->kCfxBiome
        && e->kCfxHeightPoints && e->kCfxBiomePoints;
    if (!e->cfxAvail)
        return 0;

    // Fold the new kernels into the workgroup minimum.
    cl_kernel all[] =
    {
        e->kCfxHeight, e->kCfxBiome, e->kCfxHeightPoints, e->kCfxBiomePoints
#if 0 /* parked surface-columns GPU path */
        ,
        e->kCfxColPoints
#endif
    };
    for (int i = 0; i < (int)(sizeof(all) / sizeof(*all)); i++)
    {
        size_t wgi = 0;
        e->cl.fn.GetKernelWorkGroupInfo(all[i], e->cl.device,
            CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgi), &wgi, NULL);
        if (wgi > 0 && wgi < e->localSize)
            e->localSize = wgi < GPU_LOCAL_SIZE_MAX ? wgi : GPU_LOCAL_SIZE_MAX;
    }
    return 1;
}

/*
 * Lazy fp32 build for the sparse float cfx kernels, mirroring the fp64
 * ensure's one-shot discipline (separate inited flag, no retry after a
 * failed build) with two differences: float execution needs no cl_khr_fp64,
 * so there is no extension check, and its availability is independent of
 * the fp64 side - either program failing to build leaves the other intact.
 */
static int gpuEnsureCfxFloatLocked(GpuEngine *e)
{
    if (e->cfxFloatInited)
        return e->cfxFloatAvail;
    e->cfxFloatInited = 1;
    // The float build runs on any device, so this ensure can be reached on
    // machines whose runtime never initialized (no loader, no GPU); refuse
    // there instead of calling into an unresolved loader table.
    if (!e->clAvail)
        return 0;
    e->cfxFloatProgram = gpuBuildCfxFloatProgram(e);
    if (!e->cfxFloatProgram)
        return 0;
    e->kCfxHeightPointsF = gpuKernelFromProgram(e, e->cfxFloatProgram,
        "cfxOverviewHeightPointsKernel");
    e->kCfxBiomePointsF = gpuKernelFromProgram(e, e->cfxFloatProgram,
        "cfxBiomeAtYPointsKernel");
#if 0 /* parked surface-columns GPU path */
    e->kCfxColPointsF = gpuKernelFromProgram(e, e->cfxFloatProgram,
        "cfxSurfaceColumnsPointsKernel");
#endif
    e->cfxFloatAvail = e->kCfxHeightPointsF && e->kCfxBiomePointsF;
    if (!e->cfxFloatAvail)
        return 0;

    // Fold the new kernels into the workgroup minimum.
    cl_kernel all[] =
    {
        e->kCfxHeightPointsF, e->kCfxBiomePointsF
#if 0 /* parked surface-columns GPU path */
        ,
        e->kCfxColPointsF
#endif
    };
    for (int i = 0; i < (int)(sizeof(all) / sizeof(*all)); i++)
    {
        size_t wgi = 0;
        e->cl.fn.GetKernelWorkGroupInfo(all[i], e->cl.device,
            CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgi), &wgi, NULL);
        if (wgi > 0 && wgi < e->localSize)
            e->localSize = wgi < GPU_LOCAL_SIZE_MAX ? wgi : GPU_LOCAL_SIZE_MAX;
    }
    return 1;
}

//----------------------------------------------------------------------
// Table packing from the live Generator structs

static void packPerlin(GpuTables *t, const PerlinNoise *p, int slot)
{
    memcpy(t->perm + (size_t)slot * 257, p->d, 257);
    float *info = t->pinfo + (size_t)slot * 8;
    info[0] = (float) p->a;
    info[1] = (float) p->b;
    info[2] = (float) p->c;
    info[3] = (float) p->amplitude;
    info[4] = (float) p->lacunarity;
    info[5] = (float) p->d2;
    info[6] = (float) p->t2;
    info[7] = 0.0f;
    // The cfx kernels re-run the shim's exact double chains, so they get
    // the struct values verbatim (a/b/c/amp/lac are seed-derived doubles
    // the float copy would round).
    double *infoD = t->pinfoD + (size_t)slot * 8;
    infoD[0] = p->a;
    infoD[1] = p->b;
    infoD[2] = p->c;
    infoD[3] = p->amplitude;
    infoD[4] = p->lacunarity;
    infoD[5] = p->d2;
    infoD[6] = p->t2;
    infoD[7] = p->h2;   // uint8_t; (int)info[7] in the kernel recovers it
}

static int packDoublePerlin(GpuTables *t, const DoublePerlinNoise *dp,
    int inst, int *cursor, int cap)
{
    if (inst >= GPU_INST_COUNT)
        return 1;
    if (*cursor + dp->octA.octcnt + dp->octB.octcnt > cap)
        return 1;
    t->noff[inst * 4 + 0] = *cursor;
    for (int i = 0; i < dp->octA.octcnt; i++)
        packPerlin(t, &dp->octA.octaves[i], (*cursor)++);
    t->noff[inst * 4 + 1] = dp->octA.octcnt;
    t->noff[inst * 4 + 2] = *cursor;
    for (int i = 0; i < dp->octB.octcnt; i++)
        packPerlin(t, &dp->octB.octaves[i], (*cursor)++);
    t->noff[inst * 4 + 3] = dp->octB.octcnt;
    t->namp[inst] = (float) dp->amplitude;
    t->nampD[inst] = dp->amplitude;
    return 0;
}

/*
 * Packs one spline stack into flat node arrays. Node ids: 0..ss->len-1 for
 * the stack splines (len >= 2, with typ/loc/der/children), then ss->len..
 * ss->len+ss->flen-1 for the fix splines (len == 1, value in loc[0]). The
 * arrays are pre-allocated by packTables for all three stacks.
 */
static int packSplineStack(GpuTables *t, const SplineStack *ss,
    const Spline *root, int *rootOut)
{
    int base = t->sNodeTotal;
    if (ss->len <= 0 || ss->flen < 0)
        return 1;
    int total = ss->len + ss->flen;

    for (int s = 0; s < ss->len; s++)
    {
        const Spline *sp = &ss->stack[s];
        int n = base + s;
        // getSpline() rejects len <= 0 and len >= 12; len == 1 would be a fix
        // spline, which only ever lives in fstack.
        if (sp->len <= 1 || sp->len >= 12)
            return 1;
        if (sp->typ < SP_CONTINENTALNESS || sp->typ > SP_WEIRDNESS)
            return 1;
        t->styp[n] = sp->typ;
        t->slen[n] = sp->len;
        for (int k = 0; k < 12; k++)
        {
            t->sloc[(size_t)n * 12 + k] = sp->loc[k];
            t->sder[(size_t)n * 12 + k] = sp->der[k];
            int child = -1;
            if (k < sp->len && sp->val[k])
            {
                if (sp->val[k] >= ss->stack
                    && sp->val[k] < ss->stack + ss->len)
                    child = base + (int)(sp->val[k] - ss->stack);
                else if ((const FixSpline *) sp->val[k] >= ss->fstack
                    && (const FixSpline *) sp->val[k] < ss->fstack + ss->flen)
                    child = base + ss->len
                        + (int)((const FixSpline *) sp->val[k] - ss->fstack);
                else
                    return 1;
            }
            t->schild[(size_t)n * 12 + k] = child;
        }
    }
    for (int f = 0; f < ss->flen; f++)
    {
        const FixSpline *fp = &ss->fstack[f];
        int n = base + ss->len + f;
        t->styp[n] = SP_CONTINENTALNESS;
        t->slen[n] = 1;
        for (int k = 0; k < 12; k++)
        {
            t->sloc[(size_t)n * 12 + k] = k == 0 ? fp->val : 0.0f;
            t->sder[(size_t)n * 12 + k] = 0.0f;
            t->schild[(size_t)n * 12 + k] = -1;
        }
    }
    if (root < ss->stack || root >= ss->stack + ss->len)
        return 1;
    *rootOut = base + (int)(root - ss->stack);
    t->sNodeTotal = base + total;
    return 0;
}

/*
 * Longest child chain from a spline root, iteratively: the kernel runs
 * evalSplineK with a fixed 12-frame stack, so a deeper (malformed) table
 * would write past it. Node ids are unique and 'seen' also guards against
 * cyclic child graphs. Returns -1 on allocation failure (rejected).
 */
static int packSplineDepth(const GpuTables *t, int root)
{
    int maxDepth = -1;
    int *stack = malloc(sizeof(int) * (size_t) t->sNodeTotal * 2);
    unsigned char *seen = calloc((size_t) t->sNodeTotal, 1);
    if (!stack || !seen)
    {
        free(stack);
        free(seen);
        return -1;
    }
    int sp = 0;
    stack[sp++] = root;
    stack[sp++] = 1;
    seen[root] = 1;
    while (sp)
    {
        int depth = stack[--sp];
        int node = stack[--sp];
        if (depth > maxDepth)
            maxDepth = depth;
        for (int k = 0; k < 12; k++)
        {
            int child = t->schild[(size_t) node * 12 + k];
            if (child >= 0 && !seen[child])
            {
                seen[child] = 1;
                stack[sp++] = child;
                stack[sp++] = depth + 1;
            }
        }
    }
    free(stack);
    free(seen);
    return maxDepth;
}

/*
 * Longest nonzero prefix of the selected btree's steps table: that prefix
 * bounds get_resulting_node's recursion depth, which the kernel mirrors
 * with a fixed 6-frame stack.
 */
static int packBTreeDepth(const GpuTables *t)
{
    size_t count = t->btStepsSize / sizeof(uint32_t);
    int depth = 0;
    for (size_t i = 0; i < count; i++)
    {
        if (t->btSteps[i] == 0)
            break;
        depth++;
    }
    return depth;
}

static void packBTree(GpuTables *t, int mc)
{
    // Mirrors the table selection of climateToBiome() in biomenoise.c; keep
    // the two branches in sync when adding a version.
    const uint32_t *steps;
    const int32_t (*param)[2];
    const uint64_t *nodes;
    if (mc >= MC_26_3)
    {
        steps = btree263_steps; param = btree263_param; nodes = btree263_nodes;
        t->btOrder = btree263_order;
        t->btStepsSize = sizeof(btree263_steps);
        t->btParamSize = sizeof(btree263_param);
        t->btNodesSize = sizeof(btree263_nodes);
    }
    else if (mc >= MC_26_2)
    {
        steps = btree262_steps; param = btree262_param; nodes = btree262_nodes;
        t->btOrder = btree262_order;
        t->btStepsSize = sizeof(btree262_steps);
        t->btParamSize = sizeof(btree262_param);
        t->btNodesSize = sizeof(btree262_nodes);
    }
    else if (mc >= MC_1_21_5)
    {
        steps = btree215_steps; param = btree215_param; nodes = btree215_nodes;
        t->btOrder = btree215_order;
        t->btStepsSize = sizeof(btree215_steps);
        t->btParamSize = sizeof(btree215_param);
        t->btNodesSize = sizeof(btree215_nodes);
    }
    else if (mc >= MC_1_21_4)
    {
        steps = btree21wd_steps; param = btree21wd_param;
        nodes = btree21wd_nodes;
        t->btOrder = btree21wd_order;
        t->btStepsSize = sizeof(btree21wd_steps);
        t->btParamSize = sizeof(btree21wd_param);
        t->btNodesSize = sizeof(btree21wd_nodes);
    }
    else if (mc >= MC_1_20_6)
    {
        steps = btree20_steps; param = btree20_param; nodes = btree20_nodes;
        t->btOrder = btree20_order;
        t->btStepsSize = sizeof(btree20_steps);
        t->btParamSize = sizeof(btree20_param);
        t->btNodesSize = sizeof(btree20_nodes);
    }
    else if (mc >= MC_1_19_4)
    {
        steps = btree19_steps; param = btree19_param; nodes = btree19_nodes;
        t->btOrder = btree19_order;
        t->btStepsSize = sizeof(btree19_steps);
        t->btParamSize = sizeof(btree19_param);
        t->btNodesSize = sizeof(btree19_nodes);
    }
    else if (mc >= MC_1_19_2)
    {
        steps = btree192_steps; param = btree192_param;
        nodes = btree192_nodes;
        t->btOrder = btree192_order;
        t->btStepsSize = sizeof(btree192_steps);
        t->btParamSize = sizeof(btree192_param);
        t->btNodesSize = sizeof(btree192_nodes);
    }
    else
    {
        steps = btree18_steps; param = btree18_param; nodes = btree18_nodes;
        t->btOrder = btree18_order;
        t->btStepsSize = sizeof(btree18_steps);
        t->btParamSize = sizeof(btree18_param);
        t->btNodesSize = sizeof(btree18_nodes);
    }
    t->btSteps = steps;
    t->btParam = &param[0][0];
    t->btNodes = nodes;
    t->btLen = (uint32_t)(t->btNodesSize / sizeof(uint64_t));
}

static int packTables(GpuTables *t, const BiomeNoise *bn)
{
    memset(t, 0, sizeof(*t));

    // The terrain column kernel unrolls the octave counts of the modern
    // surface noise; reject generators whose setup differs instead of
    // silently sampling the wrong octaves.
    if (bn->terrain.octmin.octcnt != 16 || bn->terrain.octmax.octcnt != 16
        || bn->terrain.octmain.octcnt != 8)
        return 1;

    int cap = 0;
    for (int i = 0; i < NP_MAX; i++)
        cap += bn->climate[i].octA.octcnt + bn->climate[i].octB.octcnt;
    cap += bn->jagged.octA.octcnt + bn->jagged.octB.octcnt;
    for (int i = 0; i < TN_MAX; i++)
        cap += bn->terrainNoise[i].octA.octcnt + bn->terrainNoise[i].octB.octcnt;
#if 0 /* parked surface-columns GPU path */
    for (int i = 0; i < AN_MAX; i++)
        cap += bn->aquiferNoise[i].octA.octcnt + bn->aquiferNoise[i].octB.octcnt;
#endif
    cap += bn->terrain.octmin.octcnt + bn->terrain.octmax.octcnt
        + bn->terrain.octmain.octcnt;
    if (cap <= 0 || cap > (1 << 16))
        return 1;

    int sNodes = bn->ss.len + bn->ss.flen
        + bn->factorSs.len + bn->factorSs.flen
        + bn->jaggedSs.len + bn->jaggedSs.flen;
    if (sNodes <= 0 || sNodes > (1 << 16))
        return 1;
    t->styp = malloc(sizeof(int) * (size_t)sNodes);
    t->slen = malloc(sizeof(int) * (size_t)sNodes);
    t->schild = malloc(sizeof(int) * (size_t)sNodes * 12);
    t->sloc = malloc(sizeof(float) * (size_t)sNodes * 12);
    t->sder = malloc(sizeof(float) * (size_t)sNodes * 12);
    t->perm = malloc((size_t)cap * 257);
    t->pinfo = malloc(sizeof(float) * (size_t)cap * 8);
    t->pinfoD = malloc(sizeof(double) * (size_t)cap * 8);
    t->pinfoF = malloc(sizeof(float) * (size_t)cap * 8);
    if (!t->styp || !t->slen || !t->schild || !t->sloc || !t->sder
        || !t->perm || !t->pinfo || !t->pinfoD || !t->pinfoF)
        return 1;

    int cursor = 0;
    int inst = 0;
    for (int i = 0; i < NP_MAX; i++, inst++)
        if (packDoublePerlin(t, &bn->climate[i], inst, &cursor, cap))
            return 1;
    if (packDoublePerlin(t, &bn->jagged, inst++, &cursor, cap))
        return 1;
    for (int i = 0; i < TN_MAX; i++, inst++)
        if (packDoublePerlin(t, &bn->terrainNoise[i], inst, &cursor, cap))
            return 1;

    t->octMinOff = cursor;
    for (int i = 0; i < bn->terrain.octmin.octcnt; i++)
        packPerlin(t, &bn->terrain.octmin.octaves[i], cursor++);
    t->octMaxOff = cursor;
    for (int i = 0; i < bn->terrain.octmax.octcnt; i++)
        packPerlin(t, &bn->terrain.octmax.octaves[i], cursor++);
    t->octMainOff = cursor;
    for (int i = 0; i < bn->terrain.octmain.octcnt; i++)
        packPerlin(t, &bn->terrain.octmain.octaves[i], cursor++);

#if 0 /* parked surface-columns GPU path */
    // The aquifer noises append after every existing instance AND after the
    // terrain octaves, so all offsets the older kernels' arguments carry
    // (instance noff entries and octMin/octMax/octMain) stay stable.
    for (int i = 0; i < AN_MAX; i++, inst++)
        if (packDoublePerlin(t, &bn->aquiferNoise[i], inst, &cursor, cap))
            return 1;
#endif
    t->octTotal = cursor;

    // The fp32 cfx kernels read float mirrors of the double tables: the
    // same packing narrowed per element (slot 7 stays h2, where the float
    // family's own pinfo pads zero - it derives h2 from info[1] instead).
    for (int i = 0; i < t->octTotal * 8; i++)
        t->pinfoF[i] = (float) t->pinfoD[i];
    for (int i = 0; i < GPU_INST_COUNT; i++)
        t->nampF[i] = (float) t->nampD[i];

    if (packSplineStack(t, &bn->ss, bn->sp, &t->rootSp))
        return 1;
    if (packSplineStack(t, &bn->factorSs, bn->factorSp, &t->rootFactor))
        return 1;
    if (packSplineStack(t, &bn->jaggedSs, bn->jaggedSp, &t->rootJagged))
        return 1;

    packBTree(t, bn->mc);
    if (t->btLen == 0 || t->btOrder == 0 || t->btOrder > 63)
        return 1;

    // Depth guards: the kernels' explicit frames cap recursion at 12
    // spline levels and 6 btree levels; reject deeper (pathological)
    // tables here so they fall back to the CPU instead of writing past
    // the frames.
    for (int i = 0; i < 3; i++)
    {
        int root = i == 0 ? t->rootSp
            : i == 1 ? t->rootFactor : t->rootJagged;
        int depth = packSplineDepth(t, root);
        if (depth < 0 || depth > 12)
            return 1;
    }
    if (packBTreeDepth(t) > 6)
        return 1;

    const SurfaceNoise *sn = &bn->terrain;
    t->snXZScale = sn->xzScale;
    t->snYScale = sn->yScale;
    t->snXZFactor = sn->xzFactor;
    t->snYFactor = sn->yFactor;
    t->snXZScaleD = sn->xzScale;
    t->snYScaleD = sn->yScale;
    t->snXZFactorD = sn->xzFactor;
    t->snYFactorD = sn->yFactor;

    t->valid = 1;
    return 0;
}

/*
 * Fingerprint of the seed-derived perm tables the kernel consumes. The
 * tables are re-derived in place on every setBiomeSeed(), so pointer
 * identity cannot detect a reseed of the same BiomeNoise. These witness
 * octaves change whenever the table state changes: a reseed re-derives every
 * channel, and a large-biomes change re-salts continentalness and erosion.
 * A collision between genuinely different states requires all four witnesses
 * to alias (~2^-4000 per pair), far below the 2^-64 chance of any 64-bit key.
 */
static uint64_t gpuProbeMix(uint64_t h, const void *data, size_t len)
{
    const uint8_t *p = data;
    while (len--)
    {
        h ^= *p++;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t gpuProbeOctave(uint64_t h, const OctaveNoise *on)
{
    int cnt = on->octcnt;
    h = gpuProbeMix(h, &cnt, sizeof(cnt));
    if (cnt > 0 && on->octaves)
        h = gpuProbeMix(h, on->octaves[0].d, sizeof(on->octaves[0].d));
    return h;
}

static uint64_t gpuTableProbe(const BiomeNoise *bn)
{
    uint64_t h = gpuProbeMix(1469598103934665603ULL, &bn->mc, sizeof(bn->mc));
    h = gpuProbeOctave(h, &bn->climate[NP_CONTINENTALNESS].octA);
    h = gpuProbeOctave(h, &bn->climate[NP_EROSION].octA);
    h = gpuProbeOctave(h, &bn->jagged.octA);
    h = gpuProbeOctave(h, &bn->terrain.octmin);
    return h;
}

/*
 * Cache-key guard for the packed tables. The key is (mc, probe): the probe
 * witnesses the seed-derived table state, and every state change a reseed or
 * a large-biomes flip can produce shows up in the probed channels. The seed
 * hash is deliberately not part of the key - it derives from the seed just
 * like the tables do, so the probe already separates the states, and the
 * height dispatch has no sha argument to offer (its entry point is
 * sha-free), which must not evict a biome dispatch's tables or vice versa.
 */
static int gpuPackForLocked(GpuEngine *e, const BiomeNoise *bn)
{
    uint64_t probe = gpuTableProbe(bn);
    if (e->haveKey && e->keyMc == bn->mc && e->keyProbe == probe)
        return e->tables.valid && !e->keyFailed;
    gpuReleasePack(e);
    e->keyMc = bn->mc;
    e->keyProbe = probe;
    e->haveKey = 1;
    if (packTables(&e->tables, bn))
    {
        // Free the partially packed set: the failure leaves buffers in
        // e->tables that only a successful pack or the next release would
        // otherwise collect.
        gpuReleasePack(e);
        e->haveKey = 1;
        e->keyFailed = 1;
        fprintf(stderr, "gpu: unsupported generator state (mc %d), "
            "falling back to CPU\n", bn->mc);
        return 0;
    }
    e->keyFailed = 0;

    e->mPerm = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)e->tables.octTotal * 257, e->tables.perm);
    e->mPinfo = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (size_t)e->tables.octTotal * 8, e->tables.pinfo);
    e->mPinfoD = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(double) * (size_t)e->tables.octTotal * 8, e->tables.pinfoD);
    e->mPinfoF = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (size_t)e->tables.octTotal * 8, e->tables.pinfoF);
    e->mNoff = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(e->tables.noff), e->tables.noff);
    e->mNamp = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(e->tables.namp), e->tables.namp);
    e->mNampD = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(e->tables.nampD), e->tables.nampD);
    e->mNampF = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(e->tables.nampF), e->tables.nampF);
    size_t nodes = (size_t)e->tables.sNodeTotal;
    if (e->mPerm && e->mPinfo && e->mPinfoD && e->mPinfoF && e->mNoff
        && e->mNamp && e->mNampD && e->mNampF)
    {
        e->mStyp = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(int) * nodes, e->tables.styp);
        e->mSlen = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(int) * nodes, e->tables.slen);
        e->mSloc = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * nodes * 12, e->tables.sloc);
        e->mSder = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * nodes * 12, e->tables.sder);
        e->mSchild = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(int) * nodes * 12, e->tables.schild);
    }
    if (e->mStyp && e->mSlen && e->mSloc && e->mSder && e->mSchild)
    {
        e->mSteps = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            e->tables.btStepsSize, (void *) e->tables.btSteps);
        e->mParam = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            e->tables.btParamSize, (void *) e->tables.btParam);
        e->mNodes = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            e->tables.btNodesSize, (void *) e->tables.btNodes);
    }
    return e->mNodes != NULL;
}

//----------------------------------------------------------------------
// Public surface

const char *gpuDeviceName(void)
{
    GPU_LOCK;
    int mode = gpuResolveModeLocked(&g_gpu);
    if (mode != GPU_MODE_OFF)
        gpuEnsureRuntimeLocked(&g_gpu);
    const char *name = g_gpu.clAvail ? g_gpu.cl.devName : "";
    GPU_UNLOCK;
    return name;
}

int gpuRuntimeInitialized(void)
{
    GPU_LOCK;
    int inited = g_gpu.clInited;
    GPU_UNLOCK;
    return inited;
}

uint64_t gpuDispatchCount(void)
{
    GPU_LOCK;
    uint64_t count = g_gpu.dispatchCount;
    GPU_UNLOCK;
    return count;
}

uint64_t gpuHeightDispatchCount(void)
{
    GPU_LOCK;
    uint64_t count = g_gpu.heightCount;
    GPU_UNLOCK;
    return count;
}

void gpuQueryMetrics(uint64_t *out)
{
    GPU_LOCK;
    memcpy(out, g_gpu.metrics.slots, sizeof(g_gpu.metrics.slots));
    // Slot 0 is not an accumulator: it reports the live queue state the
    // event timers are meaningful under.
    out[GPU_METRICS_PROFILING] = (uint64_t) (g_gpu.cl.profiling != 0);
    GPU_UNLOCK;
}

static int gpuSurfaceBiomeAvailable(const BiomeNoise *bn)
{
    if (!bn || bn->nptype >= 0 || bn->mc < MC_1_18)
        return 0;
    GPU_LOCK;
    int mode = gpuResolveModeLocked(&g_gpu);
    if (mode != GPU_MODE_OFF)
        gpuEnsureRuntimeLocked(&g_gpu);
    int healthy = g_gpu.clAvail && !g_gpu.dead;
    if (!healthy && mode == GPU_MODE_FORCE)
        gpuForceWarnLocked(&g_gpu, "no usable OpenCL device");
    GPU_UNLOCK;
    return healthy;
}

int gpuAvailable(const Generator *g)
{
    if (!g || g->dim != DIM_OVERWORLD)
        return 0;
    return gpuSurfaceBiomeAvailable(&g->bn);
}

static int gpuEnsureOutLocked(GpuEngine *e, GpuSlab *slab, int points)
{
    if (points <= slab->cap)
        return 0;
    gpuReleaseSlab(e, slab);
    slab->outNp = gpuBuf(e, CL_MEM_READ_WRITE,
        sizeof(int64_t) * (size_t)points * 6, NULL);
    if (slab->outNp)
        slab->outId = gpuBuf(e, CL_MEM_READ_WRITE,
            sizeof(int) * (size_t)points, NULL);
    if (slab->outId)
        slab->outY = gpuBuf(e, CL_MEM_READ_WRITE,
            sizeof(float) * (size_t)points, NULL);
    if (slab->outY)
        slab->inY = gpuBuf(e, CL_MEM_READ_ONLY,
            sizeof(int) * (size_t)points, NULL);
    if (slab->inY)
        slab->outYi = gpuBuf(e, CL_MEM_READ_WRITE,
            sizeof(int) * (size_t)points, NULL);
    // The sparse coord buffers ride the same growth dimension: the sparse
    // entries size them to the dispatch point count, which dominates both
    // needs (n coords for a heights chunk, nCells <= points cell origins
    // for a biomes chunk).
    if (slab->outYi)
        slab->inX = gpuBuf(e, CL_MEM_READ_ONLY,
            sizeof(int) * (size_t)points, NULL);
    if (slab->inX)
        slab->inZ = gpuBuf(e, CL_MEM_READ_ONLY,
            sizeof(int) * (size_t)points, NULL);
    if (slab->inZ)
        slab->inCX = gpuBuf(e, CL_MEM_READ_ONLY,
            sizeof(int) * (size_t)points, NULL);
    if (slab->inCX)
        slab->inCZ = gpuBuf(e, CL_MEM_READ_ONLY,
            sizeof(int) * (size_t)points, NULL);
#if 0 /* parked surface-columns GPU path */
    if (slab->inCZ)
        slab->outCol = gpuBuf(e, CL_MEM_READ_WRITE,
            sizeof(int) * (size_t)points * 4, NULL);
    if (!slab->outCol)
        return 1;
#endif
    if (!slab->inCZ)
        return 1;
    slab->cap = points;
    return 0;
}

/*
 * Profiling sink for gpuFinishDispatchN(): a dispatch that collected events
 * passes a sink with `on` set, and the read events' device timings are
 * summed into the two accumulators between the successful wait and their
 * release (the caller then adds them to the metrics under the lock).
 */
typedef struct
{
    int on;
    uint64_t execNanos;   // sum of END-START over the profiled events
    uint64_t queueNanos;  // sum of SUBMIT-QUEUED over them
} GpuReadProfile;

/*
 * Sum one completed event's device timings into the profiling accumulators:
 * END-START is the device execution span, SUBMIT-QUEUED the queue delay
 * ahead of it. A missing timestamp skips that event - metrics only, never a
 * dispatch error.
 */
static void gpuProfileEvent(cl_event ev, uint64_t *execNanos,
    uint64_t *queueNanos)
{
    cl_ulong queued = 0, submit = 0, start = 0, end = 0;
    cl_int err = g_gpu.cl.fn.GetEventProfilingInfo(ev,
        CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
    err |= g_gpu.cl.fn.GetEventProfilingInfo(ev,
        CL_PROFILING_COMMAND_SUBMIT, sizeof(submit), &submit, NULL);
    err |= g_gpu.cl.fn.GetEventProfilingInfo(ev,
        CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    err |= g_gpu.cl.fn.GetEventProfilingInfo(ev,
        CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    if (err != CL_SUCCESS)
        return;
    *execNanos += (uint64_t) (end - start);
    *queueNanos += (uint64_t) (submit - queued);
}

/*
 * Dispatch tail shared by every family and dispatch shape. `reads` carries
 * the readbacks that did enqueue in enqueue order (a batch passes one per
 * served block). Every enqueued readback is waited before its handle is
 * released, failure paths included: a CL_FALSE read drains into the
 * caller's buffer even after a later enqueue failed, so releasing it alone
 * would let the device write race the CPU fallback the caller runs on a
 * nonzero return. Each read is waited explicitly (a later event's
 * completion proves earlier ones finished, not that they finished
 * successfully), keeping a mid-queue execution failure visible like the
 * blocking reads this replaces. Any enqueue or wait error latches the
 * engine dead; the dispatch counts only once every result drained. A
 * non-NULL profiling sink collects the read events' device timings between
 * a successful wait and their release; on any error the events are only
 * released (never profiled), and NULL keeps the plain wait+release tail.
 */
static int gpuFinishDispatchN(cl_int enqErr, const cl_event *reads,
    cl_uint n, const char *what, uint64_t *counter, GpuReadProfile *prof)
{
    cl_int err = enqErr;
    cl_int waitErr = CL_SUCCESS;
    if (n)
    {
        waitErr = g_gpu.cl.fn.WaitForEvents(n, reads);
        if (err == CL_SUCCESS)
            err = waitErr;
    }
    if (prof && prof->on && waitErr == CL_SUCCESS)
    {
        for (cl_uint i = 0; i < n; i++)
            gpuProfileEvent(reads[i], &prof->execNanos, &prof->queueNanos);
    }
    for (cl_uint i = 0; i < n; i++)
        g_gpu.cl.fn.ReleaseEvent(reads[i]);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, what);
        return 1;
    }
    GPU_LOCK;
    (*counter)++;
    GPU_UNLOCK;
    return 0;
}

static int gpuFinishDispatch(cl_int enqErr, cl_event firstRead,
    cl_event lastRead, const char *what, uint64_t *counter)
{
    cl_event pending[2];
    cl_uint n = 0;
    if (firstRead)
        pending[n++] = firstRead;
    if (lastRead)
        pending[n++] = lastRead;
    return gpuFinishDispatchN(enqErr, pending, n, what, counter, NULL);
}

/*
 * Dispatch tail shared by the two sparse entries. ev[0..nEv) carries every
 * event the dispatch captured in enqueue order - the coordinate/terrainY
 * writes first (nWrites of them), then the kernel, then the read. The
 * sparse entries capture events unconditionally (the rect batch tail only
 * keeps them under the profiling opt-in) because the drain discipline here
 * needs handles: success waits only the read, whose completion proves the
 * writes and kernel drained ahead of it on the in-order queue; a failed
 * enqueue leaves the commands before it without their consumer, so the
 * failure path waits EVERY captured event before this slab can rotate back
 * into use. Either way each event is released only once waited or
 * subsumed, and any enqueue/wait error latches the engine via gpuClErr.
 * On success the four sums receive the H2D/kernel/D2H device timings and
 * queue delays (all zero without the profiling opt-in) for the caller's
 * metrics lock, and the family counter bumps once like gpuFinishDispatchN.
 */
static int gpuFinishSparseDispatch(cl_int enqErr, const cl_event *ev,
    cl_uint nEv, cl_uint nWrites, uint64_t *h2dNanos, uint64_t *kernNanos,
    uint64_t *d2hNanos, uint64_t *queueNanos, const char *what,
    uint64_t *counter)
{
    cl_int err = enqErr;
    const int prof = g_gpu.cl.profiling != 0;
    *h2dNanos = *kernNanos = *d2hNanos = *queueNanos = 0;
    if (err == CL_SUCCESS)
    {
        err = g_gpu.cl.fn.WaitForEvents(1, &ev[nEv - 1]);
        if (prof && err == CL_SUCCESS)
        {
            for (cl_uint i = 0; i < nWrites; i++)
                gpuProfileEvent(ev[i], h2dNanos, queueNanos);
            for (cl_uint i = nWrites; i + 1 < nEv; i++)
                gpuProfileEvent(ev[i], kernNanos, queueNanos);
            gpuProfileEvent(ev[nEv - 1], d2hNanos, queueNanos);
        }
    }
    else if (nEv)
    {
        g_gpu.cl.fn.WaitForEvents(nEv, ev);
    }
    for (cl_uint i = 0; i < nEv; i++)
        g_gpu.cl.fn.ReleaseEvent(ev[i]);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, what);
        return 1;
    }
    GPU_LOCK;
    (*counter)++;
    GPU_UNLOCK;
    return 0;
}

/*
 * The packed-table arguments shared by both kernels, in their common prefix
 * order. *a is the running argument index; it is advanced past the table
 * args. Returns CL_SUCCESS or the first argument error seen.
 */
static cl_int gpuSetTableArgsLocked(cl_kernel k, cl_uint *a)
{
    cl_int err = CL_SUCCESS;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mPerm);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mPinfo);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNoff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNamp);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mStyp);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSlen);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSloc);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSder);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSchild);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootSp);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootFactor);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootJagged);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSteps);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mParam);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNodes);
    cl_uint btLen = g_gpu.tables.btLen;
    cl_uint btOrder = g_gpu.tables.btOrder;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_uint), &btLen);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_uint), &btOrder);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snXZScale);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snYScale);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snXZFactor);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snYFactor);
    cl_int octMinOff = g_gpu.tables.octMinOff;
    cl_int octMaxOff = g_gpu.tables.octMaxOff;
    cl_int octMainOff = g_gpu.tables.octMainOff;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMinOff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMaxOff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMainOff);
    return err;
}

/*
 * The one guarded surface-biome dispatch: both the transparent hook
 * (gpuTrySurfaceBiomeHook) and the Generator-facing entry below funnel into
 * this single guard, so the dispatch condition exists in exactly one place.
 * The guard rejects unsupported arguments and engine states before the CPU
 * caller sees any effect; a GPU error latches the engine dead here.
 *
 * The block covers `w` columns x `rows` rows with z advancing by `zStride`
 * per row (zStride 1 = the contiguous w x rows block). The kernel runs on a
 * 2D NDRange, so a strided row set (overview consumers advancing z past the
 * queried rows) folds into ONE dispatch instead of one per row; a small
 * width no longer means a near-empty workgroup.
 */
static int gpuSurfaceBiomeDispatch(int *out, const BiomeNoise *bn,
    uint64_t sha, int scale, int x, int z, int w, int rows, int zStride,
    int64_t *np)
{
    if (!out || !bn || w <= 0 || rows <= 0 || zStride <= 0
        || (int64_t)w * rows > GPU_MAX_POINTS)
        return 1;
    if (scale != 1 && scale != 4)
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;

    GPU_LOCK;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose: CUBIOMES_GPU=off must not
    // even dlopen the OpenCL loader, let alone build programs.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu, "no usable OpenCL device");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    int points = w * rows;
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, points))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.cl.kernel;
    cl_int err = CL_SUCCESS;
    cl_uint a = 0;
    err |= gpuSetTableArgsLocked(k, &a);
    cl_int modeArg = scale == 1 ? 1 : 0;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &modeArg);
    cl_int originX = x, originZ = z, width = w, height = rows;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &originX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &originZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &width);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &height);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &zStride);
    cl_ulong shaArg = sha;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_ulong), &shaArg);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outNp);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outId);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    // Workgroups span localSize columns of one row: ceil(w/localSize) *
    // rows groups, so strided rows fill the device instead of collapsing
    // occupancy into a single group.
    size_t ls = g_gpu.localSize;
    size_t global[2] =
    {
        ((size_t) w + ls - 1) / ls * ls,
        (size_t) rows
    };
    size_t local[2] = { ls, 1 };
    cl_event evId = NULL, evNp = NULL;
    err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 2, NULL,
        global, local, 0, NULL, NULL);
    if (err == CL_SUCCESS)
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outId,
            CL_FALSE, 0, sizeof(int) * (size_t)points, out, 0, NULL, &evId);
    if (err == CL_SUCCESS && np)
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outNp,
            CL_FALSE, 0, sizeof(int64_t) * (size_t)points * 6, np, 0, NULL,
            &evNp);
    GPU_UNLOCK;
    return gpuFinishDispatch(err, evNp ? evId : NULL, evNp ? evNp : evId,
        "gpu dispatch", &g_gpu.dispatchCount);
}

int gpuTrySurfaceBiomeHook(int *out, const BiomeNoise *bn, uint64_t sha,
    int scale, int x, int z, int w, int h)
{
    return gpuSurfaceBiomeDispatch(out, bn, sha, scale, x, z, w, h, 1, NULL);
}

int gpuMapOverworldSurfaceBiome(int *out, const Generator *g, uint64_t sha,
    int scale, int x, int z, int w, int h, int64_t *np)
{
    if (!g || g->dim != DIM_OVERWORLD)
        return 1;
    return gpuSurfaceBiomeDispatch(out, &g->bn, sha, scale, x, z, w, h, 1,
        np);
}

int gpuMapOverworldSurfaceBiomeRows(int *out, const BiomeNoise *bn,
    uint64_t sha, int scale, int x, int z, int w, int rows, int zStride)
{
    return gpuSurfaceBiomeDispatch(out, bn, sha, scale, x, z, w, rows,
        zStride, NULL);
}

/*
 * The one guarded terrain-height dispatch behind gpuTryHeightHook, the
 * height-path twin of gpuSurfaceBiomeDispatch. Rejects unsupported arguments
 * and engine states, reuses the packed tables and the same guard semantics;
 * a GPU error latches the engine dead here too. Heights are quart-scale like
 * the CPU reference, so there is no scale/sha argument; ids are optional.
 * The rows/zStride block shape matches the biome dispatch (2D NDRange, one
 * dispatch per strided row set).
 */
static int gpuHeightDispatch(float *y, int *ids, const BiomeNoise *bn,
    int x, int z, int w, int rows, int zStride)
{
    if (!y || !bn || w <= 0 || rows <= 0 || zStride <= 0
        || (int64_t)w * rows > GPU_MAX_POINTS)
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;

    GPU_LOCK;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init, same as the biome dispatch.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu, "no usable OpenCL device");
        GPU_UNLOCK;
        return 1;
    }
    // Same packed tables and cache entry as the biome dispatch: one pack per
    // generator state serves both kernels.
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    int points = w * rows;
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, points))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kHeight;
    cl_int err = CL_SUCCESS;
    cl_uint a = 0;
    err |= gpuSetTableArgsLocked(k, &a);
    cl_int wantIds = ids != NULL;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &wantIds);
    cl_int originX = x, originZ = z, width = w, height = rows;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &originX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &originZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &width);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &height);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &zStride);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outY);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outId);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    size_t ls = g_gpu.localSize;
    size_t global[2] =
    {
        ((size_t) w + ls - 1) / ls * ls,
        (size_t) rows
    };
    size_t local[2] = { ls, 1 };
    cl_event evY = NULL, evId = NULL;
    err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 2, NULL,
        global, local, 0, NULL, NULL);
    if (err == CL_SUCCESS)
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outY,
            CL_FALSE, 0, sizeof(float) * (size_t)points, y, 0, NULL, &evY);
    if (err == CL_SUCCESS && ids)
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outId,
            CL_FALSE, 0, sizeof(int) * (size_t)points, ids, 0, NULL, &evId);
    GPU_UNLOCK;
    return gpuFinishDispatch(err, evId ? evY : NULL, evId ? evId : evY,
        "gpu height dispatch", &g_gpu.heightCount);
}

int gpuTryHeightHook(float *y, int *ids, const BiomeNoise *bn,
    int x, int z, int w, int h)
{
    return gpuHeightDispatch(y, ids, bn, x, z, w, h, 1);
}

int gpuMapOverworldTerrainHeightRows(float *y, int *ids, const BiomeNoise *bn,
    int x, int z, int w, int rows, int zStride)
{
    return gpuHeightDispatch(y, ids, bn, x, z, w, rows, zStride);
}

/*
 * The cfx product dispatches behind the mod's shim (confluxnative.c):
 * bit-exact double-precision re-runs of the shim's own CPU chains, unlike
 * the float approximations above. Same engine, same packed tables (the
 * double pinfo/namp copies serve these kernels), same guard discipline
 * - OFF never initializes the runtime, any GPU error latches the engine
 * dead, and a nonzero return always means "run the CPU path".
 */

/*
 * The packed-table argument prefix shared by the two cfx kernels, in their
 * common order. *a is the running argument index.
 */
static cl_int gpuSetCfxTableArgsLocked(cl_kernel k, cl_uint *a)
{
    cl_int err = CL_SUCCESS;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mPerm);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mPinfoD);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNoff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNampD);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mStyp);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSlen);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSloc);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSder);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSchild);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootSp);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootFactor);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootJagged);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSteps);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mParam);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNodes);
    cl_uint btLen = g_gpu.tables.btLen;
    cl_uint btOrder = g_gpu.tables.btOrder;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_uint), &btLen);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_uint), &btOrder);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(double),
        &g_gpu.tables.snXZScaleD);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(double),
        &g_gpu.tables.snYScaleD);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(double),
        &g_gpu.tables.snXZFactorD);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(double),
        &g_gpu.tables.snYFactorD);
    cl_int octMinOff = g_gpu.tables.octMinOff;
    cl_int octMaxOff = g_gpu.tables.octMaxOff;
    cl_int octMainOff = g_gpu.tables.octMainOff;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMinOff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMaxOff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMainOff);
    return err;
}

/*
 * The fp32 program's packed-table argument prefix: the same order as the
 * fp64 setter, with the float table mirrors (mPinfoF/mNampF) and the float
 * SurfaceNoise scalars in place of the double copies. The int buffers and
 * offsets are precision-neutral and shared with the fp64 program.
 */
static cl_int gpuSetCfxFloatTableArgsLocked(cl_kernel k, cl_uint *a)
{
    cl_int err = CL_SUCCESS;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mPerm);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mPinfoF);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNoff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNampF);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mStyp);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSlen);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSloc);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSder);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSchild);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootSp);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootFactor);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int),
        &g_gpu.tables.rootJagged);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mSteps);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mParam);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_mem), &g_gpu.mNodes);
    cl_uint btLen = g_gpu.tables.btLen;
    cl_uint btOrder = g_gpu.tables.btOrder;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_uint), &btLen);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_uint), &btOrder);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snXZScale);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snYScale);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snXZFactor);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(float),
        &g_gpu.tables.snYFactor);
    cl_int octMinOff = g_gpu.tables.octMinOff;
    cl_int octMaxOff = g_gpu.tables.octMaxOff;
    cl_int octMainOff = g_gpu.tables.octMainOff;
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMinOff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMaxOff);
    err |= g_gpu.cl.fn.SetKernelArg(k, (*a)++, sizeof(cl_int), &octMainOff);
    return err;
}

/*
 * One guarded dispatch of the shim's cfxModernOverviewHeight() over a
 * rectangular quart lattice: heights[i] receives the solver result for
 * (x4 + col*stepX4, z4 + row*stepZ4). The caller derives the lattice and
 * keeps the cheap integer interpolation host-side (the shim's macro path).
 * The detail scalars are the shim's cfxNoiseDetailForStride() result; the
 * kernel trusts them like it trusts the packed tables. Returns 0 when the
 * GPU filled all gridW*gridH heights bit-exactly and nonzero when the
 * caller must run the CPU path.
 */
int gpuCfxOverviewHeights(int *heights, const BiomeNoise *bn,
    int x4, int z4, int gridW, int gridH, int stepX4, int stepZ4,
    int mainFirst, int minMaxFirst, int truncated)
{
    if (!heights || !bn || gridW <= 0 || gridH <= 0
        || (int64_t)gridW * gridH > GPU_MAX_POINTS
        || stepX4 <= 0 || stepZ4 <= 0
        || mainFirst < 0 || mainFirst > 7
        || minMaxFirst < 0 || minMaxFirst > 15
        || (truncated != 0 && truncated != 1))
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;

    GPU_LOCK;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no fp64 OpenCL device for the cfx kernels");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    const int points = gridW * gridH;
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, points))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxHeight;
    cl_int err = CL_SUCCESS;
    cl_uint a = 0;
    err |= gpuSetCfxTableArgsLocked(k, &a);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &mainFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &minMaxFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &truncated);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &x4);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &z4);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stepX4);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stepZ4);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &gridW);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &gridH);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outYi);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    size_t ls = g_gpu.localSize;
    size_t global[2] =
    {
        ((size_t) gridW + ls - 1) / ls * ls,
        (size_t) gridH
    };
    size_t local[2] = { ls, 1 };
    cl_event ev = NULL;
    err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 2, NULL,
        global, local, 0, NULL, NULL);
    if (err == CL_SUCCESS)
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outYi,
            CL_FALSE, 0, sizeof(int) * (size_t)points, heights, 0, NULL,
            &ev);
    GPU_UNLOCK;
    return gpuFinishDispatch(err, NULL, ev, "gpu cfx height dispatch",
        &g_gpu.heightCount);
}

/*
 * One submission of gpuCfxOverviewHeights() for n independent blocks: one
 * lock acquisition, one table pack, one slab, one readback wait for the
 * whole set. Block i solves the same lattice the single entry would and
 * its heights land at heights[flatOffsets[i]] in the caller's flat layout.
 * Per-block host-side enqueue failures set bit i and skip that block's
 * kernel, read or both; the caller repairs exactly those blocks on the CPU.
 * Everything that refuses the whole batch (bad arguments, caps, engine
 * state, table pack) returns nonzero before any output write and leaves
 * *failMask alone.
 *
 * The kernel takes no output-base argument, so every block writes
 * outYi[0..points): block i's read must be enqueued before block i+1's
 * kernel, and the in-order queue runs them in exactly that order. That is
 * also why the reads stay inside the lock with the kernels - hoisting them
 * past the unlock would let another dispatcher reuse this slab first and
 * corrupt block i's results.
 */
int gpuCfxOverviewHeightsBatch(int *failMask, int *heights,
    const long *flatOffsets, const BiomeNoise *bn, const int *x4,
    const int *z4, const int *gridW, const int *gridH, const int *stepX4,
    const int *stepZ4, int n, int mainFirst, int minMaxFirst, int truncated)
{
    if (!failMask || !heights || !flatOffsets || !bn
        || !x4 || !z4 || !gridW || !gridH || !stepX4 || !stepZ4
        || n <= 0 || n > GPU_BATCH_MAX
        || mainFirst < 0 || mainFirst > 7
        || minMaxFirst < 0 || minMaxFirst > 15
        || (truncated != 0 && truncated != 1))
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;
    int64_t total = 0;
    for (int i = 0; i < n; i++)
    {
        if (gridW[i] <= 0 || gridH[i] <= 0
            || stepX4[i] <= 0 || stepZ4[i] <= 0 || flatOffsets[i] < 0)
            return 1;
        total += (int64_t) gridW[i] * gridH[i];
    }
    if (total > GPU_MAX_POINTS)
        return 1;

    // Metrics walls: the mutex wait spans the lock acquisition, the submit
    // span the guarded section up to the last enqueue. Both accumulate
    // unconditionally (counters, not event timers).
    const uint64_t mutexT0 = nowNanos();
    GPU_LOCK;
    const uint64_t submitT0 = nowNanos();
    g_gpu.metrics.slots[GPU_METRICS_MUTEX_WAIT_NANOS] += submitT0 - mutexT0;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no fp64 OpenCL device for the cfx kernels");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, (int) total))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxHeight;
    cl_uint a = 0;
    cl_int err = gpuSetCfxTableArgsLocked(k, &a);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &mainFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &minMaxFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &truncated);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }
    const cl_uint geomArg = a;

    *failMask = 0;
    cl_event reads[GPU_BATCH_MAX];
    // Kernel events exist only under the profiling opt-in (the queue
    // carries timestamps just then); the batch owns and releases them
    // itself, since the dispatch tail never sees them.
    cl_event kernels[GPU_BATCH_MAX];
    cl_uint nReads = 0;
    cl_uint nKernels = 0;
    for (int i = 0; i < n; i++)
    {
        a = geomArg;
        err = CL_SUCCESS;
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &x4[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &z4[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stepX4[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stepZ4[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &gridW[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &gridH[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem),
            &slab->outYi);
        if (err == CL_SUCCESS)
        {
            size_t ls = g_gpu.localSize;
            size_t global[2] =
            {
                ((size_t) gridW[i] + ls - 1) / ls * ls,
                (size_t) gridH[i]
            };
            size_t local[2] = { ls, 1 };
            err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 2, NULL,
                global, local, 0, NULL,
                g_gpu.cl.profiling ? &kernels[nKernels] : NULL);
        }
        if (err == CL_SUCCESS)
        {
            nKernels += g_gpu.cl.profiling ? 1u : 0u;
            g_gpu.metrics.slots[GPU_METRICS_KERNEL_LAUNCHES]++;
        }
        if (err == CL_SUCCESS)
            err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outYi,
                CL_FALSE, 0,
                sizeof(int) * ((size_t) gridW[i] * (size_t) gridH[i]),
                heights + flatOffsets[i], 0, NULL, &reads[nReads]);
        if (err != CL_SUCCESS)
            *failMask |= (int) (1u << i);
        else
        {
            nReads++;
            g_gpu.metrics.slots[GPU_METRICS_D2H_READS]++;
        }
    }
    g_gpu.metrics.slots[GPU_METRICS_BATCH_MEMBERS] += (uint64_t) n;
    g_gpu.metrics.slots[GPU_METRICS_BATCH_POINTS] += (uint64_t) total;
    g_gpu.metrics.slots[GPU_METRICS_MEMBER_HIST
        + (n < GPU_BATCH_MAX ? n : GPU_BATCH_MAX)]++;
    g_gpu.metrics.slots[GPU_METRICS_POINT_HIST
        + gpuPointHistSlot((uint64_t) total)]++;
    g_gpu.metrics.slots[GPU_METRICS_HOST_SUBMIT_NANOS]
        += nowNanos() - submitT0;
    GPU_UNLOCK;

    // The reads completing (in-order queue) proves every kernel completed,
    // so their events are profiled exactly then and released either way;
    // on a failed dispatch nothing is profiled, per gpuFinishDispatchN.
    GpuReadProfile prof = { g_gpu.cl.profiling, 0, 0 };
    const uint64_t waitT0 = nowNanos();
    const int rc = gpuFinishDispatchN(CL_SUCCESS, reads, nReads,
        "gpu cfx height batch dispatch", &g_gpu.heightCount, &prof);
    const uint64_t waited = nowNanos() - waitT0;
    uint64_t kernNanos = 0, kernQueueNanos = 0;
    if (rc == 0 && nReads > 0)
    {
        for (cl_uint i = 0; i < nKernels; i++)
            gpuProfileEvent(kernels[i], &kernNanos, &kernQueueNanos);
    }
    GPU_LOCK;
    g_gpu.metrics.slots[GPU_METRICS_HOST_WAIT_NANOS] += waited;
    if (rc == 0 && nReads > 0)
    {
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_NANOS] += kernNanos;
        g_gpu.metrics.slots[GPU_METRICS_D2H_NANOS] += prof.execNanos;
        g_gpu.metrics.slots[GPU_METRICS_QUEUE_DELAY_NANOS]
            += kernQueueNanos + prof.queueNanos;
    }
    GPU_UNLOCK;
    for (cl_uint i = 0; i < nKernels; i++)
        g_gpu.cl.fn.ReleaseEvent(kernels[i]);
    return rc;
}

/*
 * One whole sparse heights chunk in a single dispatch: n arbitrary quart
 * points (the refinement stages' waterline and cell shapes, not a
 * rectangle), one kernel launch and one readback per chunk. The
 * coordinate arrays upload per dispatch ahead of the kernel on the
 * in-order queue and the output is one contiguous [0..n) span, so the
 * launch/read counts scale with chunks, not points. Whole-or-nothing
 * rejection: the argument guard fires before the lock, and every refusal
 * after it happens before the read - the caller buffer is only ever
 * written by a dispatch that returns zero.
 */
int gpuCfxOverviewHeightsSparse(int *outY, const BiomeNoise *bn,
    const int *x4, const int *z4, int n,
    int mainFirst, int minMaxFirst, int truncated)
{
    if (!outY || !bn || !x4 || !z4
        || n < 1 || n > GPU_MAX_POINTS
        || mainFirst < 0 || mainFirst > 7
        || minMaxFirst < 0 || minMaxFirst > 15
        || (truncated != 0 && truncated != 1))
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;

    // Metrics walls, same contract as the batch entries.
    const uint64_t mutexT0 = nowNanos();
    GPU_LOCK;
    const uint64_t submitT0 = nowNanos();
    g_gpu.metrics.slots[GPU_METRICS_MUTEX_WAIT_NANOS] += submitT0 - mutexT0;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no fp64 OpenCL device for the cfx kernels");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, n))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxHeightPoints;
    cl_uint a = 0;
    cl_int err = gpuSetCfxTableArgsLocked(k, &a);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outYi);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &n);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &mainFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &minMaxFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &truncated);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    // Events are captured unconditionally (the rect entries only keep them
    // under the profiling opt-in): the sparse drain tail needs handles to
    // wait every enqueued command when a later enqueue fails.
    cl_event ev[4] = { NULL, NULL, NULL, NULL };
    cl_uint nEv = 0;
    cl_uint nWrites = 0;
    err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inX, CL_FALSE,
        0, sizeof(int) * (size_t)n, x4, 0, NULL, &ev[nEv]);
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inZ,
            CL_FALSE, 0, sizeof(int) * (size_t)n, z4, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        size_t ls = g_gpu.localSize;
        size_t global = ((size_t)n + ls - 1) / ls * ls;
        err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 1, NULL,
            &global, &ls, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_LAUNCHES]++;
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outYi,
            CL_FALSE, 0, sizeof(int) * (size_t)n, outY, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        // One packed dispatch = one member: the chunk comes from one tile.
        g_gpu.metrics.slots[GPU_METRICS_D2H_READS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_MEMBERS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_POINTS] += (uint64_t) n;
        g_gpu.metrics.slots[GPU_METRICS_MEMBER_HIST + 1]++;
        g_gpu.metrics.slots[GPU_METRICS_POINT_HIST
            + gpuPointHistSlot((uint64_t) n)]++;
    }
    g_gpu.metrics.slots[GPU_METRICS_HOST_SUBMIT_NANOS]
        += nowNanos() - submitT0;
    GPU_UNLOCK;

    uint64_t h2dNanos, kernNanos, d2hNanos, queueNanos;
    const uint64_t waitT0 = nowNanos();
    const int rc = gpuFinishSparseDispatch(err, ev, nEv, nWrites,
        &h2dNanos, &kernNanos, &d2hNanos, &queueNanos,
        "gpu cfx sparse height dispatch", &g_gpu.heightCount);
    const uint64_t waited = nowNanos() - waitT0;
    GPU_LOCK;
    g_gpu.metrics.slots[GPU_METRICS_HOST_WAIT_NANOS] += waited;
    if (rc == 0)
    {
        g_gpu.metrics.slots[GPU_METRICS_H2D_NANOS] += h2dNanos;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_NANOS] += kernNanos;
        g_gpu.metrics.slots[GPU_METRICS_D2H_NANOS] += d2hNanos;
        g_gpu.metrics.slots[GPU_METRICS_QUEUE_DELAY_NANOS] += queueNanos;
    }
    GPU_UNLOCK;
    return rc;
}

/*
 * The fp32 twin of gpuCfxOverviewHeightsSparse(): one whole sparse heights
 * chunk in a single dispatch, served by the float program's points kernel.
 * Request shape, whole-or-nothing refusal, slab buffers, drain tail and
 * metrics slots are identical to the fp64 entry; the kernel executes the
 * SAME cfx expression compiled to single precision, so outputs approximate
 * the CPU reference coordinate-dependently and are NOT bit-exact - the
 * fp64 entry stays the exact reference. The float program needs no
 * cl_khr_fp64 device and its availability is independent of the fp64 side.
 */
int gpuCfxOverviewHeightsSparseFloat(int *outY, const BiomeNoise *bn,
    const int *x4, const int *z4, int n,
    int mainFirst, int minMaxFirst, int truncated)
{
    if (!outY || !bn || !x4 || !z4
        || n < 1 || n > GPU_MAX_POINTS
        || mainFirst < 0 || mainFirst > 7
        || minMaxFirst < 0 || minMaxFirst > 15
        || (truncated != 0 && truncated != 1))
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;

    // Metrics walls, same contract as the batch entries.
    const uint64_t mutexT0 = nowNanos();
    GPU_LOCK;
    const uint64_t submitT0 = nowNanos();
    g_gpu.metrics.slots[GPU_METRICS_MUTEX_WAIT_NANOS] += submitT0 - mutexT0;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxFloatLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no float cfx kernel program");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, n))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxHeightPointsF;
    cl_uint a = 0;
    cl_int err = gpuSetCfxFloatTableArgsLocked(k, &a);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outYi);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &n);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &mainFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &minMaxFirst);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &truncated);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    // Events are captured unconditionally (the rect entries only keep them
    // under the profiling opt-in): the sparse drain tail needs handles to
    // wait every enqueued command when a later enqueue fails.
    cl_event ev[4] = { NULL, NULL, NULL, NULL };
    cl_uint nEv = 0;
    cl_uint nWrites = 0;
    err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inX, CL_FALSE,
        0, sizeof(int) * (size_t)n, x4, 0, NULL, &ev[nEv]);
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inZ,
            CL_FALSE, 0, sizeof(int) * (size_t)n, z4, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        size_t ls = g_gpu.localSize;
        size_t global = ((size_t)n + ls - 1) / ls * ls;
        err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 1, NULL,
            &global, &ls, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_LAUNCHES]++;
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outYi,
            CL_FALSE, 0, sizeof(int) * (size_t)n, outY, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        // One packed dispatch = one member: the chunk comes from one tile.
        g_gpu.metrics.slots[GPU_METRICS_D2H_READS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_MEMBERS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_POINTS] += (uint64_t) n;
        g_gpu.metrics.slots[GPU_METRICS_MEMBER_HIST + 1]++;
        g_gpu.metrics.slots[GPU_METRICS_POINT_HIST
            + gpuPointHistSlot((uint64_t) n)]++;
    }
    g_gpu.metrics.slots[GPU_METRICS_HOST_SUBMIT_NANOS]
        += nowNanos() - submitT0;
    GPU_UNLOCK;

    uint64_t h2dNanos, kernNanos, d2hNanos, queueNanos;
    const uint64_t waitT0 = nowNanos();
    const int rc = gpuFinishSparseDispatch(err, ev, nEv, nWrites,
        &h2dNanos, &kernNanos, &d2hNanos, &queueNanos,
        "gpu cfx sparse float height dispatch", &g_gpu.heightCount);
    const uint64_t waited = nowNanos() - waitT0;
    GPU_LOCK;
    g_gpu.metrics.slots[GPU_METRICS_HOST_WAIT_NANOS] += waited;
    if (rc == 0)
    {
        g_gpu.metrics.slots[GPU_METRICS_H2D_NANOS] += h2dNanos;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_NANOS] += kernNanos;
        g_gpu.metrics.slots[GPU_METRICS_D2H_NANOS] += d2hNanos;
        g_gpu.metrics.slots[GPU_METRICS_QUEUE_DELAY_NANOS] += queueNanos;
    }
    GPU_UNLOCK;
    return rc;
}

/*
 * One submission of gpuCfxSurfaceBiomes() for n independent caller-Y
 * blocks: one lock acquisition, one table pack, one slab, one readback
 * wait for the whole set. Block i resolves the same voronoi/climate chain
 * the single entry would and its ids land at ids[flatOffsets[i]] in the
 * caller's flat layout; its w[i]*rows[i]-int terrainY plane sits at the
 * same offset in the caller's flat terrainY, which uploads as ONE write
 * into the slab's input buffer before the kernels enqueue - the in-order
 * queue runs the write ahead of every kernel, so no event tracking is
 * needed beyond the reads (the metrics opt-in attaches a profiling event
 * to the write, owned like the kernel events). Each kernel resolves its
 * plane through its
 * per-block inYBase input offset, the flat layout's counterpart of the
 * interleaved output reads. Per-block host-side enqueue failures set bit i
 * and skip that block's kernel, read or both; the caller repairs exactly
 * those blocks on the CPU. Everything that refuses the whole batch (bad
 * arguments, caps, engine state, table pack, slab growth, the input
 * upload itself) returns nonzero before any output write and leaves
 * *failMask alone.
 *
 * The kernel takes no output-base argument, so every block writes
 * outId[0..points): block i's read must be enqueued before block i+1's
 * kernel, and the in-order queue runs them in exactly that order. That is
 * also why the reads stay inside the lock with the kernels - hoisting them
 * past the unlock would let another dispatcher reuse this slab first and
 * corrupt block i's results.
 */
int gpuCfxSurfaceBiomesBatch(int *failMask, int *ids, const long *flatOffsets,
    const BiomeNoise *bn, uint64_t sha,
    const int *blockX, const int *blockZ, const int *w, const int *rows,
    const int *stride, const int *terrainY, int n)
{
    if (!failMask || !ids || !flatOffsets || !bn
        || !blockX || !blockZ || !w || !rows || !stride || !terrainY
        || n <= 0 || n > GPU_BATCH_MAX)
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;
    int64_t total = 0;
    for (int i = 0; i < n; i++)
    {
        if (w[i] <= 0 || rows[i] <= 0 || stride[i] <= 0 || flatOffsets[i] < 0)
            return 1;
        total += (int64_t) w[i] * rows[i];
    }
    if (total > GPU_MAX_POINTS)
        return 1;

    // Metrics walls, same contract as the height batch above.
    const uint64_t mutexT0 = nowNanos();
    GPU_LOCK;
    const uint64_t submitT0 = nowNanos();
    g_gpu.metrics.slots[GPU_METRICS_MUTEX_WAIT_NANOS] += submitT0 - mutexT0;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no fp64 OpenCL device for the cfx kernels");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, (int) total))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxBiome;
    cl_uint a = 0;
    cl_int err = gpuSetCfxTableArgsLocked(k, &a);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }
    const cl_uint geomArg = a;

    // The one packed terrainY upload; its event is the batch's only H2D
    // measurement and, like the kernel events, is owned and released here.
    cl_event write = NULL;
    err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inY, CL_FALSE,
        0, sizeof(int) * (size_t)total, terrainY, 0, NULL,
        g_gpu.cl.profiling ? &write : NULL);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clEnqueueWriteBuffer");
        GPU_UNLOCK;
        return 1;
    }
    g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;

    *failMask = 0;
    cl_event reads[GPU_BATCH_MAX];
    cl_event kernels[GPU_BATCH_MAX];  // profiling only, see the height batch
    cl_uint nReads = 0;
    cl_uint nKernels = 0;
    for (int i = 0; i < n; i++)
    {
        a = geomArg;
        err = CL_SUCCESS;
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &blockX[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &blockZ[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &w[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &rows[i]);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stride[i]);
        cl_ulong shaArg = sha;
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_ulong), &shaArg);
        cl_int inYBase = (cl_int) flatOffsets[i];
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &inYBase);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inY);
        err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outId);
        if (err == CL_SUCCESS)
        {
            size_t ls = g_gpu.localSize;
            size_t global[2] =
            {
                ((size_t) w[i] + ls - 1) / ls * ls,
                (size_t) rows[i]
            };
            size_t local[2] = { ls, 1 };
            err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 2, NULL,
                global, local, 0, NULL,
                g_gpu.cl.profiling ? &kernels[nKernels] : NULL);
        }
        if (err == CL_SUCCESS)
        {
            nKernels += g_gpu.cl.profiling ? 1u : 0u;
            g_gpu.metrics.slots[GPU_METRICS_KERNEL_LAUNCHES]++;
        }
        if (err == CL_SUCCESS)
            err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outId,
                CL_FALSE, 0,
                sizeof(int) * ((size_t) w[i] * (size_t) rows[i]),
                ids + flatOffsets[i], 0, NULL, &reads[nReads]);
        if (err != CL_SUCCESS)
            *failMask |= (int) (1u << i);
        else
        {
            nReads++;
            g_gpu.metrics.slots[GPU_METRICS_D2H_READS]++;
        }
    }
    g_gpu.metrics.slots[GPU_METRICS_BATCH_MEMBERS] += (uint64_t) n;
    g_gpu.metrics.slots[GPU_METRICS_BATCH_POINTS] += (uint64_t) total;
    g_gpu.metrics.slots[GPU_METRICS_MEMBER_HIST
        + (n < GPU_BATCH_MAX ? n : GPU_BATCH_MAX)]++;
    g_gpu.metrics.slots[GPU_METRICS_POINT_HIST
        + gpuPointHistSlot((uint64_t) total)]++;
    g_gpu.metrics.slots[GPU_METRICS_HOST_SUBMIT_NANOS]
        += nowNanos() - submitT0;
    GPU_UNLOCK;

    // Kernel/write events are profiled after the reads completed (the
    // in-order queue ran the write ahead of every kernel), see the height
    // batch.
    GpuReadProfile prof = { g_gpu.cl.profiling, 0, 0 };
    const uint64_t waitT0 = nowNanos();
    const int rc = gpuFinishDispatchN(CL_SUCCESS, reads, nReads,
        "gpu cfx biome batch dispatch", &g_gpu.dispatchCount, &prof);
    const uint64_t waited = nowNanos() - waitT0;
    uint64_t kernNanos = 0, kernQueueNanos = 0, writeNanos = 0,
        writeQueueNanos = 0;
    if (rc == 0 && nReads > 0)
    {
        for (cl_uint i = 0; i < nKernels; i++)
            gpuProfileEvent(kernels[i], &kernNanos, &kernQueueNanos);
        if (write)
            gpuProfileEvent(write, &writeNanos, &writeQueueNanos);
    }
    GPU_LOCK;
    g_gpu.metrics.slots[GPU_METRICS_HOST_WAIT_NANOS] += waited;
    if (rc == 0 && nReads > 0)
    {
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_NANOS] += kernNanos;
        g_gpu.metrics.slots[GPU_METRICS_H2D_NANOS] += writeNanos;
        g_gpu.metrics.slots[GPU_METRICS_D2H_NANOS] += prof.execNanos;
        g_gpu.metrics.slots[GPU_METRICS_QUEUE_DELAY_NANOS]
            += kernQueueNanos + writeQueueNanos + prof.queueNanos;
    }
    GPU_UNLOCK;
    for (cl_uint i = 0; i < nKernels; i++)
        g_gpu.cl.fn.ReleaseEvent(kernels[i]);
    if (write)
        g_gpu.cl.fn.ReleaseEvent(write);
    return rc;
}

/*
 * One whole sparse biomes chunk in a single dispatch: nCells cells of
 * edge*edge stride-spaced points, the cell origins and the packed
 * cell-major terrainY uploading ahead of one kernel launch, one readback
 * of the contiguous [0..nCells*edge*edge) id span. The kernel derives
 * each point's coordinates from its flat index and its cell's origin, so
 * the caller needs no per-point output offsets. Whole-or-nothing refusal
 * like the heights twin; the per-point voronoi/climate chain is the same
 * expression the rect entry runs.
 */
int gpuCfxSurfaceBiomesSparse(int *outIds, const BiomeNoise *bn, uint64_t sha,
    const int *cellX, const int *cellZ,
    const int *terrainY, int edge, int stride, int nCells)
{
    if (!outIds || !bn || !cellX || !cellZ || !terrainY
        || edge < 1 || stride < 1 || nCells < 1
        || edge > GPU_MAX_POINTS || nCells > GPU_MAX_POINTS
        || (int64_t) nCells * (int64_t) edge * (int64_t) edge
            > GPU_MAX_POINTS)
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;
    const int total = nCells * edge * edge;

    // Metrics walls, same contract as the batch entries.
    const uint64_t mutexT0 = nowNanos();
    GPU_LOCK;
    const uint64_t submitT0 = nowNanos();
    g_gpu.metrics.slots[GPU_METRICS_MUTEX_WAIT_NANOS] += submitT0 - mutexT0;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no fp64 OpenCL device for the cfx kernels");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, total))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxBiomePoints;
    cl_uint a = 0;
    cl_int err = gpuSetCfxTableArgsLocked(k, &a);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inCX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inCZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inY);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outId);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &total);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &edge);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stride);
    cl_ulong shaArg = sha;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_ulong), &shaArg);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    cl_event ev[5] = { NULL, NULL, NULL, NULL, NULL };
    cl_uint nEv = 0;
    cl_uint nWrites = 0;
    err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inCX, CL_FALSE,
        0, sizeof(int) * (size_t)nCells, cellX, 0, NULL, &ev[nEv]);
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inCZ,
            CL_FALSE, 0, sizeof(int) * (size_t)nCells, cellZ, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inY,
            CL_FALSE, 0, sizeof(int) * (size_t)total, terrainY, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        size_t ls = g_gpu.localSize;
        size_t global = ((size_t)total + ls - 1) / ls * ls;
        err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 1, NULL,
            &global, &ls, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_LAUNCHES]++;
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outId,
            CL_FALSE, 0, sizeof(int) * (size_t)total, outIds, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        // One packed dispatch = one member: the chunk comes from one tile.
        g_gpu.metrics.slots[GPU_METRICS_D2H_READS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_MEMBERS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_POINTS] += (uint64_t) total;
        g_gpu.metrics.slots[GPU_METRICS_MEMBER_HIST + 1]++;
        g_gpu.metrics.slots[GPU_METRICS_POINT_HIST
            + gpuPointHistSlot((uint64_t) total)]++;
    }
    g_gpu.metrics.slots[GPU_METRICS_HOST_SUBMIT_NANOS]
        += nowNanos() - submitT0;
    GPU_UNLOCK;

    uint64_t h2dNanos, kernNanos, d2hNanos, queueNanos;
    const uint64_t waitT0 = nowNanos();
    const int rc = gpuFinishSparseDispatch(err, ev, nEv, nWrites,
        &h2dNanos, &kernNanos, &d2hNanos, &queueNanos,
        "gpu cfx sparse biome dispatch", &g_gpu.dispatchCount);
    const uint64_t waited = nowNanos() - waitT0;
    GPU_LOCK;
    g_gpu.metrics.slots[GPU_METRICS_HOST_WAIT_NANOS] += waited;
    if (rc == 0)
    {
        g_gpu.metrics.slots[GPU_METRICS_H2D_NANOS] += h2dNanos;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_NANOS] += kernNanos;
        g_gpu.metrics.slots[GPU_METRICS_D2H_NANOS] += d2hNanos;
        g_gpu.metrics.slots[GPU_METRICS_QUEUE_DELAY_NANOS] += queueNanos;
    }
    GPU_UNLOCK;
    return rc;
}

/*
 * The fp32 twin of gpuCfxSurfaceBiomesSparse(): one whole sparse biomes
 * chunk in a single dispatch, served by the float program's points kernel.
 * Request shape, whole-or-nothing refusal, slab buffers, drain tail and
 * metrics slots are identical to the fp64 entry; the per-point voronoi/
 * climate chain is the SAME expression compiled to single precision, so
 * ids approximate the CPU reference coordinate-dependently and are NOT
 * bit-exact - the fp64 entry stays the exact reference. No fp64 device
 * requirement; availability is independent of the fp64 side.
 */
int gpuCfxSurfaceBiomesSparseFloat(int *outIds, const BiomeNoise *bn,
    uint64_t sha, const int *cellX, const int *cellZ,
    const int *terrainY, int edge, int stride, int nCells)
{
    if (!outIds || !bn || !cellX || !cellZ || !terrainY
        || edge < 1 || stride < 1 || nCells < 1
        || edge > GPU_MAX_POINTS || nCells > GPU_MAX_POINTS
        || (int64_t) nCells * (int64_t) edge * (int64_t) edge
            > GPU_MAX_POINTS)
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;
    const int total = nCells * edge * edge;

    // Metrics walls, same contract as the batch entries.
    const uint64_t mutexT0 = nowNanos();
    GPU_LOCK;
    const uint64_t submitT0 = nowNanos();
    g_gpu.metrics.slots[GPU_METRICS_MUTEX_WAIT_NANOS] += submitT0 - mutexT0;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxFloatLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no float cfx kernel program");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, total))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxBiomePointsF;
    cl_uint a = 0;
    cl_int err = gpuSetCfxFloatTableArgsLocked(k, &a);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inCX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inCZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inY);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outId);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &total);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &edge);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stride);
    cl_ulong shaArg = sha;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_ulong), &shaArg);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    cl_event ev[5] = { NULL, NULL, NULL, NULL, NULL };
    cl_uint nEv = 0;
    cl_uint nWrites = 0;
    err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inCX, CL_FALSE,
        0, sizeof(int) * (size_t)nCells, cellX, 0, NULL, &ev[nEv]);
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inCZ,
            CL_FALSE, 0, sizeof(int) * (size_t)nCells, cellZ, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inY,
            CL_FALSE, 0, sizeof(int) * (size_t)total, terrainY, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        size_t ls = g_gpu.localSize;
        size_t global = ((size_t)total + ls - 1) / ls * ls;
        err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 1, NULL,
            &global, &ls, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_LAUNCHES]++;
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outId,
            CL_FALSE, 0, sizeof(int) * (size_t)total, outIds, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        // One packed dispatch = one member: the chunk comes from one tile.
        g_gpu.metrics.slots[GPU_METRICS_D2H_READS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_MEMBERS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_POINTS] += (uint64_t) total;
        g_gpu.metrics.slots[GPU_METRICS_MEMBER_HIST + 1]++;
        g_gpu.metrics.slots[GPU_METRICS_POINT_HIST
            + gpuPointHistSlot((uint64_t) total)]++;
    }
    g_gpu.metrics.slots[GPU_METRICS_HOST_SUBMIT_NANOS]
        += nowNanos() - submitT0;
    GPU_UNLOCK;

    uint64_t h2dNanos, kernNanos, d2hNanos, queueNanos;
    const uint64_t waitT0 = nowNanos();
    const int rc = gpuFinishSparseDispatch(err, ev, nEv, nWrites,
        &h2dNanos, &kernNanos, &d2hNanos, &queueNanos,
        "gpu cfx sparse float biome dispatch", &g_gpu.dispatchCount);
    const uint64_t waited = nowNanos() - waitT0;
    GPU_LOCK;
    g_gpu.metrics.slots[GPU_METRICS_HOST_WAIT_NANOS] += waited;
    if (rc == 0)
    {
        g_gpu.metrics.slots[GPU_METRICS_H2D_NANOS] += h2dNanos;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_NANOS] += kernNanos;
        g_gpu.metrics.slots[GPU_METRICS_D2H_NANOS] += d2hNanos;
        g_gpu.metrics.slots[GPU_METRICS_QUEUE_DELAY_NANOS] += queueNanos;
    }
    GPU_UNLOCK;
    return rc;
}

#if 0 /* parked: surface-columns GPU path - net-wall negative in the product seam per the seeded e2e A/B (opening wall +6.2%, first tile +76%); restore = flip the parked guards in both repos and rebuild */
/*
 * Shared body of the two surface-columns point entries (fp64 exact and
 * fp32 drift-budgeted): n anchors, the coordinate buffers in, one kernel
 * launch, one interleaved readback into a private staging buffer, then the
 * plane de-interleave into the caller's four arrays. The staging keeps the
 * readback at exactly one transfer (one D2H count per dispatch, like the
 * other sparse entries) without forcing the caller's planes together; it
 * is private to this dispatch, so no slab rotation can race the
 * de-interleave. Whole-or-nothing: every refusal below fires before the
 * first enqueue, and the caller's arrays are only touched after the
 * readback drained successfully. The aquifer hash seeds travel as per-seed
 * kernel arguments, read from the BiomeNoise at dispatch time - the pack
 * cache deliberately stays (mc, probe), and a reseed re-uploads the seeds
 * with the tables it already re-keys. Columns are height-plane outputs, so
 * the family counter is the height one.
 */
static int gpuCfxColumnsDispatch(int *outSolidY, int *outFluidY,
    int *outSurfaceY, int *outFlags, const BiomeNoise *bn,
    const int *blockX, const int *blockZ, int n, int wantFloat)
{
    if (!outSolidY || !outFluidY || !outSurfaceY || !outFlags
        || !bn || !blockX || !blockZ
        || n < 1 || n > GPU_MAX_POINTS)
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;

    // Metrics walls, same contract as the batch entries.
    const uint64_t mutexT0 = nowNanos();
    GPU_LOCK;
    const uint64_t submitT0 = nowNanos();
    g_gpu.metrics.slots[GPU_METRICS_MUTEX_WAIT_NANOS] += submitT0 - mutexT0;
    int mode = gpuResolveModeLocked(&g_gpu);
    // OFF precedes the runtime init on purpose, same as the float paths.
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = wantFloat
        ? gpuEnsureCfxFloatLocked(&g_gpu) : gpuEnsureCfxLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : wantFloat
                        ? "no float cfx kernel program"
                        : "no fp64 OpenCL device for the cfx kernels");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, n))
    {
        GPU_UNLOCK;
        return 1;
    }
    // Private staging for the interleaved readback: allocated before the
    // first enqueue so an allocation failure is a plain whole-chunk
    // refusal, and freed on every path below.
    int *staging = malloc(sizeof(int) * (size_t) n * 4);
    if (!staging)
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = wantFloat ? g_gpu.kCfxColPointsF : g_gpu.kCfxColPoints;
    cl_uint a = 0;
    cl_int err = wantFloat
        ? gpuSetCfxFloatTableArgsLocked(k, &a)
        : gpuSetCfxTableArgsLocked(k, &a);
    cl_ulong seedLoArg = bn->aquiferSeedLo;
    cl_ulong seedHiArg = bn->aquiferSeedHi;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_ulong), &seedLoArg);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_ulong), &seedHiArg);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outCol);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &n);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        free(staging);
        GPU_UNLOCK;
        return 1;
    }

    // Events are captured unconditionally (the rect entries only keep them
    // under the profiling opt-in): the sparse drain tail needs handles to
    // wait every enqueued command when a later enqueue fails.
    cl_event ev[4] = { NULL, NULL, NULL, NULL };
    cl_uint nEv = 0;
    cl_uint nWrites = 0;
    err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inX, CL_FALSE,
        0, sizeof(int) * (size_t) n, blockX, 0, NULL, &ev[nEv]);
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inZ,
            CL_FALSE, 0, sizeof(int) * (size_t) n, blockZ, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        nWrites++;
        g_gpu.metrics.slots[GPU_METRICS_H2D_WRITES]++;
        size_t ls = g_gpu.localSize;
        size_t global = ((size_t) n + ls - 1) / ls * ls;
        err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 1, NULL,
            &global, &ls, 0, NULL, &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_LAUNCHES]++;
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outCol,
            CL_FALSE, 0, sizeof(int) * (size_t) n * 4, staging, 0, NULL,
            &ev[nEv]);
    }
    if (err == CL_SUCCESS)
    {
        nEv++;
        // One packed dispatch = one member: the chunk comes from one tile.
        g_gpu.metrics.slots[GPU_METRICS_D2H_READS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_MEMBERS]++;
        g_gpu.metrics.slots[GPU_METRICS_BATCH_POINTS] += (uint64_t) n;
        g_gpu.metrics.slots[GPU_METRICS_MEMBER_HIST + 1]++;
        g_gpu.metrics.slots[GPU_METRICS_POINT_HIST
            + gpuPointHistSlot((uint64_t) n)]++;
    }
    g_gpu.metrics.slots[GPU_METRICS_HOST_SUBMIT_NANOS]
        += nowNanos() - submitT0;
    GPU_UNLOCK;

    uint64_t h2dNanos, kernNanos, d2hNanos, queueNanos;
    const uint64_t waitT0 = nowNanos();
    const int rc = gpuFinishSparseDispatch(err, ev, nEv, nWrites,
        &h2dNanos, &kernNanos, &d2hNanos, &queueNanos,
        wantFloat
            ? "gpu cfx sparse float columns dispatch"
            : "gpu cfx sparse columns dispatch",
        &g_gpu.heightCount);
    const uint64_t waited = nowNanos() - waitT0;
    GPU_LOCK;
    g_gpu.metrics.slots[GPU_METRICS_HOST_WAIT_NANOS] += waited;
    if (rc == 0)
    {
        g_gpu.metrics.slots[GPU_METRICS_H2D_NANOS] += h2dNanos;
        g_gpu.metrics.slots[GPU_METRICS_KERNEL_NANOS] += kernNanos;
        g_gpu.metrics.slots[GPU_METRICS_D2H_NANOS] += d2hNanos;
        g_gpu.metrics.slots[GPU_METRICS_QUEUE_DELAY_NANOS] += queueNanos;
    }
    GPU_UNLOCK;
    if (rc == 0)
    {
        for (int i = 0; i < n; i++)
        {
            outSolidY[i] = staging[i * 4 + 0];
            outFluidY[i] = staging[i * 4 + 1];
            outSurfaceY[i] = staging[i * 4 + 2];
            outFlags[i] = staging[i * 4 + 3];
        }
    }
    free(staging);
    return rc;
}

int gpuCfxSurfaceColumnsPoints(int *outSolidY, int *outFluidY,
    int *outSurfaceY, int *outFlags, const BiomeNoise *bn,
    const int *blockX, const int *blockZ, int n)
{
    return gpuCfxColumnsDispatch(outSolidY, outFluidY, outSurfaceY, outFlags,
        bn, blockX, blockZ, n, 0);
}

int gpuCfxSurfaceColumnsPointsFloat(int *outSolidY, int *outFluidY,
    int *outSurfaceY, int *outFlags, const BiomeNoise *bn,
    const int *blockX, const int *blockZ, int n)
{
    return gpuCfxColumnsDispatch(outSolidY, outFluidY, outSurfaceY, outFlags,
        bn, blockX, blockZ, n, 1);
}
#endif /* parked surface-columns GPU path */

/*
 * One guarded dispatch of the shim's cfxSurfaceBiomes() inner loop: for
 * each point (blockX + col*stride, blockZ + row*stride) the caller's
 * terrain height (clamped at 62) picks the voronoi nucleus, and the
 * double-precision climate stack resolves the biome id. terrainY is
 * row-major w*rows and uploads per call; ids receives w*rows biome ids.
 * The nucleus-memoized CPU path this replaces is itself bit-identical to
 * the plain loop, so bit-exactness against the kernel is preserved. The
 * caller decides the minimum batch size (dispatch overhead dominates tiny
 * refine cells); this guard only rejects technically unsupported shapes.
 * Returns 0 when the GPU filled all ids bit-exactly and nonzero when the
 * caller must run the CPU path.
 */
int gpuCfxSurfaceBiomes(int *ids, const BiomeNoise *bn, uint64_t sha,
    int blockX, int blockZ, int w, int rows, int stride,
    const int *terrainY)
{
    if (!ids || !terrainY || !bn || w <= 0 || rows <= 0 || stride <= 0
        || (int64_t)w * rows > GPU_MAX_POINTS)
        return 1;
    if (bn->nptype >= 0 || bn->mc < MC_1_18)
        return 1;

    GPU_LOCK;
    int mode = gpuResolveModeLocked(&g_gpu);
    if (mode == GPU_MODE_OFF)
    {
        GPU_UNLOCK;
        return 1;
    }
    gpuEnsureRuntimeLocked(&g_gpu);
    const int cfxReady = gpuEnsureCfxLocked(&g_gpu);
    if (!g_gpu.clAvail || g_gpu.dead || !cfxReady)
    {
        if (mode == GPU_MODE_FORCE)
            gpuForceWarnLocked(&g_gpu,
                !g_gpu.clAvail || g_gpu.dead
                    ? "no usable OpenCL device"
                    : "no fp64 OpenCL device for the cfx kernels");
        GPU_UNLOCK;
        return 1;
    }
    if (!gpuPackForLocked(&g_gpu, bn))
    {
        GPU_UNLOCK;
        return 1;
    }
    const int points = w * rows;
    GpuSlab *slab = &g_gpu.slabs[g_gpu.activeSlab];
    g_gpu.activeSlab ^= 1;
    if (gpuEnsureOutLocked(&g_gpu, slab, points))
    {
        GPU_UNLOCK;
        return 1;
    }

    cl_kernel k = g_gpu.kCfxBiome;
    cl_int err = CL_SUCCESS;
    cl_uint a = 0;
    err |= gpuSetCfxTableArgsLocked(k, &a);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &blockX);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &blockZ);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &w);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &rows);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &stride);
    cl_ulong shaArg = sha;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_ulong), &shaArg);
    cl_int inYBase = 0;
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_int), &inYBase);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->inY);
    err |= g_gpu.cl.fn.SetKernelArg(k, a++, sizeof(cl_mem), &slab->outId);
    if (err != CL_SUCCESS)
    {
        gpuClErr(&g_gpu, err, "clSetKernelArg");
        GPU_UNLOCK;
        return 1;
    }

    err = g_gpu.cl.fn.EnqueueWriteBuffer(g_gpu.cl.queue, slab->inY, CL_TRUE,
        0, sizeof(int) * (size_t)points, terrainY, 0, NULL, NULL);
    cl_event ev = NULL;
    if (err == CL_SUCCESS)
    {
        size_t ls = g_gpu.localSize;
        size_t global[2] =
        {
            ((size_t) w + ls - 1) / ls * ls,
            (size_t) rows
        };
        size_t local[2] = { ls, 1 };
        err = g_gpu.cl.fn.EnqueueNDRangeKernel(g_gpu.cl.queue, k, 2, NULL,
            global, local, 0, NULL, NULL);
    }
    if (err == CL_SUCCESS)
        err = g_gpu.cl.fn.EnqueueReadBuffer(g_gpu.cl.queue, slab->outId,
            CL_FALSE, 0, sizeof(int) * (size_t)points, ids, 0, NULL, &ev);
    GPU_UNLOCK;
    return gpuFinishDispatch(err, NULL, ev, "gpu cfx biome dispatch",
        &g_gpu.dispatchCount);
}
