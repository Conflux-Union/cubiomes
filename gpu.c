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

#define GPU_INST_COUNT (NP_MAX + 1 + TN_MAX)
// Dispatch size cap: ~1M points keeps one output slab bounded (~59 MB worst
// case with the climate-np plane; two rotating slabs ~120 MB); larger
// queries take the CPU fallback and stay correct.
#define GPU_MAX_POINTS (1 << 20)
#define GPU_CL_OPTS "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt"
#define GPU_LOCAL_SIZE_MAX 128

typedef struct
{
    int valid;
    int octTotal;
    uint8_t *perm;   // octTotal * 257
    float *pinfo;    // octTotal * 8 (float copy, float kernels)
    double *pinfoD;  // octTotal * 8 (double copy, cfx kernels); slot 7 = h2
    int noff[GPU_INST_COUNT * 4];
    float namp[GPU_INST_COUNT];
    double nampD[GPU_INST_COUNT];
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
    int cap;       // points all five buffers are sized for
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
    cl_kernel kHeight;  // terrainHeightKernel, created from the built program
    cl_kernel kCfxHeight, kCfxBiome;  // cfxOverviewHeightKernel, cfxBiomeAtYKernel
    int cfxAvail;       // fp64 program built + both cfx kernels live
    int cfxInited;      // the one lazy cfx build attempt has been made
    size_t localSize;
    GpuTables tables;
    uint64_t keyProbe;
    int keyMc;
    int haveKey;
    int keyFailed;
    uint64_t dispatchCount;
    uint64_t heightCount;
    cl_mem mPerm, mPinfo, mNoff, mNamp;
    cl_mem mPinfoD, mNampD;
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
    // clInited, clAvail, dead, forceWarned, cl, kHeight, cfxAvail,
    // localSize (re-set at first init), tables, cache key, counters,
    // slabs.
};

static void gpuClErr(GpuEngine *e, cl_int err, const char *what)
{
    fprintf(stderr, "gpu: %s failed (OpenCL error %d); engine disabled\n",
        what, (int) err);
    e->dead = 1;
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
        &e->mPinfoD, &e->mNampD,
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
 * Builds the fp64 cfx program from the pragma + cfx source section. A
 * build failure prints the compiler log once and reports unavailable, so
 * a device that advertises the extension but cannot compile the kernels
 * degrades to the CPU path like any other engine problem.
 */
static cl_program gpuBuildCfxProgram(GpuEngine *e)
{
    const char *parts[2] = { GPU_CL_PRAGMA_FP64, GPU_CL_SRC_CFX };
    const size_t lens[2] =
    {
        strlen(GPU_CL_PRAGMA_FP64),
        strlen(GPU_CL_SRC_CFX),
    };
    cl_int err = CL_SUCCESS;
    cl_program program = e->cl.fn.CreateProgramWithSource(e->cl.context, 2,
        parts, lens, &err);
    if (!program || err != CL_SUCCESS)
        return NULL;
    err = e->cl.fn.BuildProgram(program, 1, &e->cl.device, GPU_CL_OPTS,
        NULL, NULL);
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
                fprintf(stderr, "gpu: cfx fp64 program build failed:\n%s\n",
                    log);
                free(log);
            }
        }
        e->cl.fn.ReleaseProgram(program);
        return NULL;
    }
    return program;
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
    if (!gpuDeviceHasFp64(&e->cl))
        return 0;
    e->cfxProgram = gpuBuildCfxProgram(e);
    if (!e->cfxProgram)
        return 0;
    e->kCfxHeight = gpuKernelFromProgram(e, e->cfxProgram,
        "cfxOverviewHeightKernel");
    e->kCfxBiome = gpuKernelFromProgram(e, e->cfxProgram,
        "cfxBiomeAtYKernel");
    e->cfxAvail = e->kCfxHeight && e->kCfxBiome;
    if (!e->cfxAvail)
        return 0;

    // Fold the new kernels into the workgroup minimum.
    cl_kernel all[] = { e->kCfxHeight, e->kCfxBiome };
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
    if (!t->styp || !t->slen || !t->schild || !t->sloc || !t->sder
        || !t->perm || !t->pinfo || !t->pinfoD)
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
    t->octTotal = cursor;

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
    e->mNoff = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(e->tables.noff), e->tables.noff);
    e->mNamp = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(e->tables.namp), e->tables.namp);
    e->mNampD = gpuBuf(e, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(e->tables.nampD), e->tables.nampD);
    size_t nodes = (size_t)e->tables.sNodeTotal;
    if (e->mPerm && e->mPinfo && e->mPinfoD && e->mNoff && e->mNamp
        && e->mNampD)
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
    if (!slab->outYi)
        return 1;
    slab->cap = points;
    return 0;
}

/*
 * Dispatch tail shared by all four families. `enqErr` carries any enqueue
 * failure; the events are the readbacks that did enqueue (the last one is
 * set whenever all enqueues succeeded). Every enqueued readback is waited
 * before its handle is released, failure paths included: a CL_FALSE read
 * drains into the caller's buffer even after a later enqueue fails, so
 * releasing it alone would let the device write race the CPU fallback the
 * caller runs on a nonzero return. The earlier readback is waited
 * explicitly (the last event's completion proves it finished, not that it
 * finished successfully), keeping a mid-queue execution failure visible
 * like the blocking reads this replaces. Any enqueue or wait error latches
 * the engine dead; the dispatch counts only once every result drained.
 */
static int gpuFinishDispatch(cl_int enqErr, cl_event firstRead,
    cl_event lastRead, const char *what, uint64_t *counter)
{
    cl_event pending[2];
    cl_uint n = 0;
    if (firstRead)
        pending[n++] = firstRead;
    if (lastRead)
        pending[n++] = lastRead;
    cl_int err = enqErr;
    if (n)
    {
        cl_int waitErr = g_gpu.cl.fn.WaitForEvents(n, pending);
        if (err == CL_SUCCESS)
            err = waitErr;
    }
    for (cl_uint i = 0; i < n; i++)
        g_gpu.cl.fn.ReleaseEvent(pending[i]);
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
