/*
 * CPU-vs-GPU drift-budget harness for the 1.18+ Overworld surface biome and
 * terrain-height GPU engine (gpu.c). Build and run via `make gpu-test`.
 *
 * The GPU engine evaluates the CPU's double-precision reference chain in
 * float, so its outputs drift in a coordinate-dependent way (negligible near
 * the origin, growing toward the world border). The harness therefore gates
 * two separate things:
 *
 * - Determinism contracts, held bit-exact: the harness's CPU replication
 *   against the real wrapper (pure CPU-vs-CPU), GPU-vs-GPU query-shape
 *   invariance (row dispatch vs isolated calls compute identical results),
 *   the CPU body in off mode, public-entry engagement counts, and that
 *   CUBIOMES_GPU=off never initializes the OpenCL runtime.
 * - Approximation quality, gated by per-bucket drift budgets: biome-id flip
 *   rate, per-channel climate |delta| (np units), terrain-height |delta|
 *   (blocks), and an explicit water/land flip count (height crossing sea
 *   level 62). Buckets split by |origin|: near-field (<=100k), 1M-class and
 *   the 30M world border. `--report` runs the matrix without budget
 *   assertions to re-measure; without it, exceeding any budget exits nonzero.
 *
 * The CPU side is the reference itself: the real mapOverworldSurfaceBiome()
 * wrapper guards the harness's own replication (which additionally exposes
 * the climate np values), and the GPU dispatch is compared against it.
 * Without a usable OpenCL device it skips cleanly with exit 0.
 *
 * Matrix: the default run covers one seed over all origins (positive,
 * negative and zero-straddling) and shapes on the main version, a reduced
 * origin set for every supported biome-tree version, and one large-biomes
 * case. `--full` extends the main-version sweep to all seeds and both
 * world-border extremes (quart ±7.5M) on every version. Each case prints
 * its header before it runs and its timing after, so a stall names the case
 * and the phase that caused it.
 *
 * Height sections exercise the terrain overview heights the same way: the
 * guarded dispatch (gpuTryHeightHook) against the CPU reference over the
 * consumer query shapes (dense quart maps, the mod's strided overview rows,
 * 4x4 chunk tiles, isolated single columns), the transparent
 * mapOverworldTerrainHeight()/mapApproxHeight() entries in off/auto/force
 * with engagement counting, and a row-vs-isolated shape-invariance case.
 *
 * A transparent-path section exercises the public mapOverworldSurfaceBiome()
 * entry (the guarded GPU hook) in off/auto/force modes: off must reproduce
 * the CPU reference exactly, auto/force must engage the engine on every
 * public call (dispatch counter, CUBIOMES_ENABLE_GPU builds) and their
 * outputs are budgeted like the guarded dispatch.
 */

#include "gpu.h"

#include "biomenoise.h"
#include "biomes.h"
#include "generator.h"
#include "layers.h"
#include "rng.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DENSE_W 96
#define DENSE_H 96
#define STRIDED_W 96
#define STRIDED_H 8
#define STRIDED_STRIDE 16 // quart stride = 64 blocks, as the mod's overview rows

#define NP_CH 6

#define SEA_LEVEL_Y 62.0f // heights below this read as water in the mod

typedef struct
{
    int scale;      // 1 or 4
    int stride;     // 1 = dense, > 1 = strided rows in this scale's units
    int w, h;
} Shape;

static const Shape SHAPES[] =
{
    {4, 1, DENSE_W, DENSE_H},                       // scale 4 dense
    {1, 1, DENSE_W, DENSE_H},                       // scale 1 dense (voronoi)
    {4, STRIDED_STRIDE, STRIDED_W, STRIDED_H},      // scale 4 strided rows
};
#define SHAPE_CNT (int)(sizeof(SHAPES) / sizeof(SHAPES[0]))

static const char *shapeNameByIdx(int idx)
{
    static const char *names[] = { "scale4-dense", "scale1-dense",
        "scale4-strided" };
    return names[idx];
}

static double nowSeconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static int64_t npDelta(int64_t a, int64_t b)
{
    return a > b ? a - b : b - a;
}

//----------------------------------------------------------------------
// Drift accounting: per-distance-bucket accumulators + budgets.

enum { BK_NEAR, BK_MID, BK_BORDER, BK_CNT };
static const char *bucketNames[BK_CNT] =
    { "near<=100k", "1M-class", "30M-border" };

static int bucketOf(int ox, int oz)
{
    long long ax = ox < 0 ? -(long long) ox : ox;
    long long az = oz < 0 ? -(long long) oz : oz;
    long long m = ax > az ? ax : az;
    if (m <= 100000)
        return BK_NEAR;
    if (m <= 1500000)
        return BK_MID;
    return BK_BORDER;
}

typedef struct
{
    long long pts;
    long long idMismatch;
    long long npMismatch[NP_CH];
    double npAbsSum[NP_CH];
    int64_t npMax[NP_CH];
    long long heightPts;
    long long heightDeltaPts;
    double heightAbsSum;
    float heightMaxDelta;
    long long waterLandFlips;
} Bucket;

static Bucket buckets[BK_CNT];

static void accountIds(int bucket, int gpuId, int refId)
{
    buckets[bucket].pts++;
    if (gpuId != refId)
        buckets[bucket].idMismatch++;
}

static void accountNp(int bucket, const int64_t *gnp, const int64_t *cnp)
{
    Bucket *b = &buckets[bucket];
    for (int k = 0; k < NP_CH; k++)
    {
        if (gnp[k] != cnp[k])
        {
            int64_t d = npDelta(gnp[k], cnp[k]);
            b->npMismatch[k]++;
            b->npAbsSum[k] += (double) d;
            if (d > b->npMax[k])
                b->npMax[k] = d;
        }
    }
}

static void accountHeight(int bucket, float gpuY, float refY)
{
    Bucket *b = &buckets[bucket];
    b->heightPts++;
    if (gpuY != refY)
    {
        float d = fabsf(gpuY - refY);
        b->heightDeltaPts++;
        b->heightAbsSum += (double) d;
        if (d > b->heightMaxDelta)
            b->heightMaxDelta = d;
    }
    if ((gpuY >= SEA_LEVEL_Y) != (refY >= SEA_LEVEL_Y))
        b->waterLandFlips++;
}

/*
 * Drift budgets per bucket. Rates are fractions of compared points; deltas
 * are maxima (np units for climate channels, blocks for heights). Derived
 * from the phase-1 measurement (365077 pts, report mode): rates carry an
 * order of magnitude of headroom; height deltas carry 5x capped by the
 * physical [-64, 320] output range. A broken structure (wrong channel,
 * mangled state machine) produces order-of-magnitude rate flips, not
 * budget-sized drift. Measured values live in task-6-float-report.md.
 */
typedef struct
{
    double maxIdFlipRate;
    double maxWaterLandFlipRate;
    float maxHeightDelta;
    int64_t maxNpDelta;
} Budget;

static const Budget BUDGETS[BK_CNT] =
{
    /* near <=100k: 0.0171% id flips, 0.0139% wl flips, dY 5, np 625 */
    { 0.0017, 0.0014, 25.0f, 6500 },
    /* 1M-class: 0.1849% id flips, 0.2600% wl flips, dY 23, np 935 */
    { 0.0185, 0.026, 115.0f, 9400 },
    /* 30M border: 0.8442% id flips, 1.2450% wl flips, dY 77, np 1553 */
    { 0.085, 0.125, 192.0f, 16000 },
};

static int reportBudgets(int assert)
{
    int fail = 0;
    printf("\ndrift budgets %s:\n", assert ? "(asserted)" : "(report only)");
    printf("%-11s %9s %13s %13s %12s %13s %12s %14s\n",
        "bucket", "pts", "id flips", "id flip rate", "wl flips",
        "wl flip rate", "max dY", "max np delta");
    for (int b = 0; b < BK_CNT; b++)
    {
        const Bucket *t = &buckets[b];
        int64_t npMaxAll = 0;
        for (int k = 0; k < NP_CH; k++)
            if (t->npMax[k] > npMaxAll)
                npMaxAll = t->npMax[k];
        double idRate = t->pts ? (double) t->idMismatch / t->pts : 0.0;
        double wlRate = t->heightPts
            ? (double) t->waterLandFlips / t->heightPts : 0.0;
        printf("%-11s %9lld %13lld %12.4f%% %12lld %12.4f%% %12.1f %14lld\n",
            bucketNames[b], t->pts, t->idMismatch, idRate * 100.0,
            t->waterLandFlips, wlRate * 100.0,
            (double) t->heightMaxDelta, (long long) npMaxAll);
        if (!assert)
            continue;
        if (idRate > BUDGETS[b].maxIdFlipRate)
        {
            printf("  BUDGET FAIL: id flip rate %.5f%% > %.5f%%\n",
                idRate * 100.0, BUDGETS[b].maxIdFlipRate * 100.0);
            fail = 1;
        }
        if (wlRate > BUDGETS[b].maxWaterLandFlipRate)
        {
            printf("  BUDGET FAIL: water/land flip rate %.5f%% > %.5f%%\n",
                wlRate * 100.0, BUDGETS[b].maxWaterLandFlipRate * 100.0);
            fail = 1;
        }
        if (t->heightMaxDelta > BUDGETS[b].maxHeightDelta)
        {
            printf("  BUDGET FAIL: max height delta %.1f > %.1f\n",
                (double) t->heightMaxDelta,
                (double) BUDGETS[b].maxHeightDelta);
            fail = 1;
        }
        for (int k = 0; k < NP_CH; k++)
        {
            if (t->npMax[k] > BUDGETS[b].maxNpDelta)
            {
                printf("  BUDGET FAIL: np[%d] max delta %lld > %lld\n",
                    k, (long long) t->npMax[k],
                    (long long) BUDGETS[b].maxNpDelta);
                fail = 1;
            }
        }
    }
    // per-channel climate deltas (np units), mean over all points + max
    printf("%-11s", "np |delta|");
    for (int k = 0; k < NP_CH; k++)
    {
        double sum = 0.0;
        int64_t mx = 0;
        long long pts = 0;
        for (int b = 0; b < BK_CNT; b++)
        {
            sum += buckets[b].npAbsSum[k];
            pts += buckets[b].pts;
            if (buckets[b].npMax[k] > mx)
                mx = buckets[b].npMax[k];
        }
        printf("  np[%d] mean %.2f max %lld%s\n", k,
            pts ? sum / pts : 0.0, (long long) mx,
            k == 4 ? " (depth)" : "");
    }
    {
        double sum = 0.0;
        float mx = 0.0f;
        long long pts = 0, nonzero = 0;
        for (int b = 0; b < BK_CNT; b++)
        {
            sum += buckets[b].heightAbsSum;
            pts += buckets[b].heightPts;
            nonzero += buckets[b].heightDeltaPts;
            if (buckets[b].heightMaxDelta > mx)
                mx = buckets[b].heightMaxDelta;
        }
        printf("height |delta|: mean %.3f max %.1f over %lld pts "
            "(%lld nonzero)\n", pts ? sum / pts : 0.0, (double) mx,
            pts, nonzero);
    }
    fflush(stdout);
    return fail;
}

//----------------------------------------------------------------------
// CPU reference paths (both pinned off-GPU; pure CPU-vs-CPU golden).

/*
 * CPU height reference. mapOverworldTerrainHeight() is transparently
 * GPU-accelerated when CUBIOMES_ENABLE_GPU is compiled in, so the reference
 * pins the engine off around the call to stay a pure-CPU golden.
 */
static double cpuHeightRef(float *y, int *ids, const BiomeNoise *bn,
    int x, int z, int w, int h)
{
    const int modeBefore = gpuGetMode();
    gpuSetMode(GPU_MODE_OFF);
    double t0 = nowSeconds();
    mapOverworldTerrainHeight(y, ids, bn, x, z, w, h);
    double t1 = nowSeconds();
    gpuSetMode(modeBefore);
    return t1 - t0;
}

/*
 * Replication of mapOverworldSurfaceBiome() that also exposes the climate
 * np values per point. Uses the same public primitives as the wrapper
 * (mapOverworldTerrainHeight, floordiv, voronoiAccess3D, sampleBiomeNoise).
 */
static void cpuSurfaceBiomeRef(int *out, int64_t *np, const BiomeNoise *bn,
    uint64_t sha, int scale, int x, int z, int w, int h)
{
    if (scale == 4)
    {
        float *height = malloc(sizeof(float) * (size_t)w * h);
        if (!height)
        {
            fprintf(stderr, "gpu-test: out of memory\n");
            exit(2);
        }
        cpuHeightRef(height, 0, bn, x, z, w, h);
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                int surface = (int) height[j*w+i];
                if (surface < 62) surface = 62;
                int64_t *pnp = np ? np + (size_t)(j*w+i) * NP_CH : NULL;
                out[j*w+i] = sampleBiomeNoise(bn, pnp,
                    x+i, floordiv(surface, 4), z+j, 0, 0);
            }
        }
        free(height);
        return;
    }

    const int x4min = floordiv(x, 4);
    const int z4min = floordiv(z, 4);
    const int x4max = floordiv(x + w - 1, 4) + 1;
    const int z4max = floordiv(z + h - 1, 4) + 1;
    const int sw = x4max - x4min + 1;
    const int sh = z4max - z4min + 1;
    float *height = malloc(sizeof(float) * (size_t)sw * sh);
    if (!height)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }
    cpuHeightRef(height, 0, bn, x4min, z4min, sw, sh);
    for (int j = 0; j < h; j++)
    {
        const int blockZ = z + j;
        const int baseZ = floordiv(blockZ, 4);
        const int fz = blockZ - baseZ * 4;
        const int iz = baseZ - z4min;
        for (int i = 0; i < w; i++)
        {
            const int blockX = x + i;
            const int baseX = floordiv(blockX, 4);
            const int fx = blockX - baseX * 4;
            const int ix = baseX - x4min;
            const int h00 = (int)height[iz*sw + ix];
            const int h10 = (int)height[iz*sw + ix+1];
            const int h01 = (int)height[(iz+1)*sw + ix];
            const int h11 = (int)height[(iz+1)*sw + ix+1];
            const int top = h00 + floordiv((h10-h00)*fx, 4);
            const int bottom = h01 + floordiv((h11-h01)*fx, 4);
            int surface = top + floordiv((bottom-top)*fz, 4);
            if (surface < 62) surface = 62;
            int x4, y4, z4;
            voronoiAccess3D(sha, blockX, surface, blockZ, &x4, &y4, &z4);
            int64_t *pnp = np ? np + (size_t)(j*w+i) * NP_CH : NULL;
            out[j*w+i] = sampleBiomeNoise(bn, pnp, x4, y4, z4, 0, 0);
        }
    }
    free(height);
}

//----------------------------------------------------------------------
// Guarded-dispatch matrix.

typedef struct
{
    long long pts;
    long long idMismatch;
    long long npMismatch;
    long long wrapperReplicationMismatch;
    double tWrap, tRef, tGpu;
} CaseStats;

static void reportCase(const CaseStats *st)
{
    printf("  -> %s: %lld pts, id drift %lld, np drift %lld "
        "[wrap %.1fs ref %.1fs gpu %.1fs]\n",
        st->wrapperReplicationMismatch ? "FAIL" :
            (st->idMismatch || st->npMismatch) ? "drift" : "ok",
        st->pts, st->idMismatch, st->npMismatch,
        st->tWrap, st->tRef, st->tGpu);
    fflush(stdout);
}

/*
 * Runs one (version, seed, origin, shape) case:
 * per row (dense = a single row), compare
 *   - harness reference replication vs the real wrapper (self-check,
 *     must stay bit-exact: both sides are pure CPU)
 *   - GPU dispatch vs the reference (ids + np, budgeted)
 */
static void runCase(int mc, uint32_t flags, uint64_t seed, int ox, int oz,
    int shapeIdx, CaseStats *st, int *fail)
{
    const Shape *sh = &SHAPES[shapeIdx];
    const int bucket = bucketOf(ox, oz);
    printf("case mc=%d seed=%llu origin=(%d,%d) %s [%s]\n",
        mc, (unsigned long long) seed, ox, oz, shapeNameByIdx(shapeIdx),
        bucketNames[bucket]);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);
    if (g.bn.nptype >= 0)
    {
        fprintf(stderr, "gpu-test: generator for mc %d not initialized\n", mc);
        exit(2);
    }

    const int stride = sh->stride;
    const int rawW = (sh->w - 1) * stride + 1;
    int *wrapIds = malloc(sizeof(int) * (size_t)rawW);
    int *refIds = malloc(sizeof(int) * (size_t)rawW);
    int *gpuIds = malloc(sizeof(int) * (size_t)rawW);
    int64_t *refNp = malloc(sizeof(int64_t) * (size_t)rawW * NP_CH);
    int64_t *gpuNp = malloc(sizeof(int64_t) * (size_t)rawW * NP_CH);
    if (!wrapIds || !refIds || !gpuIds || !refNp || !gpuNp)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }

    for (int j = 0; j < sh->h; j++)
    {
        const int rowZ = oz + j * stride;
        double t0 = nowSeconds();
        // The real public wrapper over the dense row in off mode: the pure
        // CPU body. The replication check pins the harness's own CPU
        // reference against it, so a reference regression shows up here.
        const int modeBefore = gpuGetMode();
        gpuSetMode(GPU_MODE_OFF);
        mapOverworldSurfaceBiome(wrapIds, &g.bn, g.sha, sh->scale,
            ox, rowZ, rawW, 1);
        gpuSetMode(modeBefore);
        double t1 = nowSeconds();
        // harness reference (must match the wrapper bit-exactly)
        cpuSurfaceBiomeRef(refIds, refNp, &g.bn, g.sha, sh->scale,
            ox, rowZ, rawW, 1);
        double t2 = nowSeconds();
        // GPU dispatch over the same dense row
        if (gpuMapOverworldSurfaceBiome(gpuIds, &g, g.sha, sh->scale,
                ox, rowZ, rawW, 1, gpuNp))
        {
            fprintf(stderr, "gpu-test: GPU dispatch failed at mc=%d "
                "seed=%llu (%d,%d) - engine reported unavailable\n",
                mc, (unsigned long long) seed, ox, rowZ);
            exit(2);
        }
        double t3 = nowSeconds();
        st->tWrap += t1 - t0;
        st->tRef += t2 - t1;
        st->tGpu += t3 - t2;

        for (int i = 0; i < sh->w; i++)
        {
            const int idx = i * stride;
            st->pts++;
            if (refIds[idx] != wrapIds[idx])
            {
                st->wrapperReplicationMismatch++;
                if (st->wrapperReplicationMismatch <= 3)
                    printf("  ref-vs-wrapper @(%d,%d): ref=%d wrapper=%d\n",
                        ox + i * stride, rowZ, refIds[idx], wrapIds[idx]);
            }
            accountIds(bucket, gpuIds[idx], refIds[idx]);
            if (gpuIds[idx] != refIds[idx])
            {
                st->idMismatch++;
                if (st->idMismatch <= 3)
                    printf("  id @(%d,%d): gpu=%d cpu=%d\n",
                        ox + i * stride, rowZ, gpuIds[idx], refIds[idx]);
            }
            accountNp(bucket, gpuNp + (size_t)idx * NP_CH,
                refNp + (size_t)idx * NP_CH);
            for (int k = 0; k < NP_CH; k++)
            {
                const int64_t *cnp = refNp + (size_t)idx * NP_CH;
                const int64_t *gnp = gpuNp + (size_t)idx * NP_CH;
                if (gnp[k] != cnp[k])
                    st->npMismatch++;
            }
        }
    }

    free(wrapIds);
    free(refIds);
    free(gpuIds);
    free(refNp);
    free(gpuNp);
}

static void runAndAccount(int mc, uint32_t flags, uint64_t seed, int ox,
    int oz, int shapeIdx, CaseStats *total, int *fail)
{
    CaseStats st = {0};
    runCase(mc, flags, seed, ox, oz, shapeIdx, &st, fail);
    reportCase(&st);
    if (st.wrapperReplicationMismatch)
        *fail = 1;
    total->pts += st.pts;
    total->idMismatch += st.idMismatch;
    total->npMismatch += st.npMismatch;
    total->wrapperReplicationMismatch += st.wrapperReplicationMismatch;
}

//----------------------------------------------------------------------
// Transparent dispatch path: the public mapOverworldSurfaceBiome() entry,
// which tries the guarded GPU hook before its CPU body when compiled with
// CUBIOMES_ENABLE_GPU. The off pass runs the CPU body and must reproduce the
// reference bit-exactly; auto/force run the GPU and are budgeted. auto/force
// must engage the engine on every public call (dispatch counter), off must
// not touch it. Points are counted once per case even though all three
// passes compare them.

static const Shape TSHAPES[] =
{
    {4, 1, 64, 8},                                  // surface scale 4 dense
    {1, 1, 64, 8},                                  // surface scale 1 dense
    {4, STRIDED_STRIDE, STRIDED_W, 2},              // surface scale 4 strided
};

static const char *tshapeNameByIdx(int idx)
{
    static const char *names[] = { "surface4-dense", "surface1-dense",
        "surface4-strided" };
    return names[idx];
}

static void runTransparentCase(int mc, uint32_t flags, uint64_t seed, int ox,
    int oz, int shapeIdx, CaseStats *total, int *fail)
{
    const Shape *sh = &TSHAPES[shapeIdx];
    const int bucket = bucketOf(ox, oz);
    printf("transparent case mc=%d seed=%llu origin=(%d,%d) %s [%s]\n",
        mc, (unsigned long long) seed, ox, oz, tshapeNameByIdx(shapeIdx),
        bucketNames[bucket]);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);

    const int stride = sh->stride;
    const int rawW = (sh->w - 1) * stride + 1;
    int *refIds = malloc(sizeof(int) * (size_t)rawW);
    int *pubIds = malloc(sizeof(int) * (size_t)rawW);
    int64_t *refNp = malloc(sizeof(int64_t) * (size_t)rawW * NP_CH);
    if (!refIds || !pubIds || !refNp)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }

    static const int modes[] =
        { GPU_MODE_OFF, GPU_MODE_AUTO, GPU_MODE_FORCE };
    static const char *modeNames[] = { "off", "auto", "force" };
    const int modeBefore = gpuGetMode();
    long long idDrift = 0, idHard = 0;
    int shapeFail = 0;

    for (int p = 0; p < 3; p++)
    {
        gpuSetMode(modes[p]);
        uint64_t before = gpuDispatchCount();
        for (int j = 0; j < sh->h; j++)
        {
            const int rowZ = oz + j * stride;
            cpuSurfaceBiomeRef(refIds, refNp, &g.bn, g.sha, sh->scale,
                ox, rowZ, rawW, 1);
            mapOverworldSurfaceBiome(pubIds, &g.bn, g.sha, sh->scale,
                ox, rowZ, rawW, 1);
            for (int i = 0; i < rawW; i++)
            {
                if (pubIds[i] != refIds[i])
                {
                    if (modes[p] == GPU_MODE_OFF)
                    {
                        // CPU body: regression, never drift
                        idHard++;
                        if (idHard <= 3)
                            printf("  %s id @(%d,%d): pub=%d ref=%d\n",
                                modeNames[p], ox + i, rowZ, pubIds[i],
                                refIds[i]);
                    }
                    else
                    {
                        idDrift++;
                    }
                }
                if (modes[p] != GPU_MODE_OFF)
                    accountIds(bucket, pubIds[i], refIds[i]);
            }
        }
        uint64_t engaged = gpuDispatchCount() - before;
        uint64_t expected = 0;
#ifdef CUBIOMES_ENABLE_GPU
        if (modes[p] != GPU_MODE_OFF)
            expected = (uint64_t) sh->h;
        if (engaged != expected)
        {
            fprintf(stderr, "gpu-test: public entry %s mode engaged %llu/%llu"
                " dispatches (mc=%d seed=%llu)\n", modeNames[p],
                (unsigned long long) engaged, (unsigned long long) expected,
                mc, (unsigned long long) seed);
            shapeFail = 1;
        }
#endif
        printf("  -> %s: hard mismatch %lld, drift %lld [gpu %llu/%llu]\n",
            modeNames[p], idHard, idDrift,
            (unsigned long long) (modes[p] != GPU_MODE_OFF ? engaged : 0),
            (unsigned long long) (modes[p] != GPU_MODE_OFF ? expected : 0));
        fflush(stdout);
    }
    gpuSetMode(modeBefore);

    free(refIds);
    free(pubIds);
    free(refNp);
    total->pts += (long long) sh->w * sh->h;
    if (idHard || shapeFail)
        *fail = 1;
}

/*
 * Query-shape invariance through the public entry: one strided/dense row
 * computed by a single GPU dispatch must equal isolated single-point public
 * queries (each its own dispatch) bit-exactly - the kernel's per-point work
 * is independent of the query shape. Against the CPU reference the row is
 * budgeted like any other GPU output.
 */
static void runTransparentShapeCase(int mc, uint32_t flags, uint64_t seed,
    int ox, int oz, CaseStats *total, int *fail)
{
    const int bucket = bucketOf(ox, oz);
    printf("transparent shape case mc=%d seed=%llu origin=(%d,%d) "
        "row-vs-isolated [%s]\n",
        mc, (unsigned long long) seed, ox, oz, bucketNames[bucket]);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);

    const int modeBefore = gpuGetMode();
    gpuSetMode(GPU_MODE_AUTO);
    const uint64_t countBefore = gpuDispatchCount();
    uint64_t calls = 0;
    long long invarianceMismatch = 0;
    long long refDrift = 0;
    int shapeFail = 0;
    long long pts = 0;

    for (int s = 0; s < 2; s++)
    {
        const int scale = s == 0 ? 4 : 1;
        const int w = scale == 4
            ? (STRIDED_W - 1) * STRIDED_STRIDE + 1
            : 64;
        const int step = scale == 4 ? STRIDED_STRIDE : 7;
        int *row = malloc(sizeof(int) * (size_t)w);
        int *refIds = malloc(sizeof(int) * (size_t)w);
        int64_t *refNp = malloc(sizeof(int64_t) * (size_t)w * NP_CH);
        if (!row || !refIds || !refNp)
        {
            fprintf(stderr, "gpu-test: out of memory\n");
            exit(2);
        }
        cpuSurfaceBiomeRef(refIds, refNp, &g.bn, g.sha, scale, ox, oz, w, 1);
        mapOverworldSurfaceBiome(row, &g.bn, g.sha, scale, ox, oz, w, 1);
        calls++;
        long long scaleInvariance = 0;
        for (int i = 0; i < w; i += step)
        {
            int one;
            mapOverworldSurfaceBiome(&one, &g.bn, g.sha, scale,
                ox + i, oz, 1, 1);
            calls++;
            // GPU-vs-GPU: identical by construction, any difference is a
            // determinism defect, not drift
            if (one != row[i])
            {
                scaleInvariance++;
                if (scaleInvariance <= 3)
                    printf("  scale%d isolated @(%d,%d): one=%d row=%d\n",
                        scale, ox + i, oz, one, row[i]);
            }
            accountIds(bucket, row[i], refIds[i]);
            if (row[i] != refIds[i])
                refDrift++;
            pts++;
        }
        printf("  -> scale%d: %lld sampled pts, invariance mismatch %lld, "
            "ref drift %lld\n", scale, pts, scaleInvariance, refDrift);
        fflush(stdout);
        invarianceMismatch += scaleInvariance;
        free(row);
        free(refIds);
        free(refNp);
    }

#ifdef CUBIOMES_ENABLE_GPU
    uint64_t engaged = gpuDispatchCount() - countBefore;
    if (engaged != calls)
    {
        fprintf(stderr, "gpu-test: shape case engaged %llu/%llu dispatches"
            " (mc=%d)\n", (unsigned long long) engaged,
            (unsigned long long) calls, mc);
        shapeFail = 1;
    }
#else
    (void) countBefore;
    (void) calls;
#endif
    gpuSetMode(modeBefore);

    total->pts += pts;
    if (invarianceMismatch || shapeFail)
        *fail = 1;
}

/*
 * Reseed regression for the table cache: the engine keys its packed tables
 * on a fingerprint of the seed-derived state, because setBiomeSeed()
 * re-derives the tables in place and pointer identity cannot see a reseed.
 * Public-entry calls on the SAME BiomeNoise across a reseed and across a
 * large-biomes flip (same seed, same sha) must each match the CPU reference
 * (budgeted); a stale cache would repeat the first configuration's tables.
 */
static void runTransparentReseedCase(int mc, uint32_t flags, uint64_t seedA,
    uint64_t seedB, CaseStats *total, int *fail)
{
    printf("transparent reseed case mc=%d seeds %llu -> %llu -> %llu(large)\n",
        mc, (unsigned long long) seedA, (unsigned long long) seedB,
        (unsigned long long) seedA);
    fflush(stdout);

    Generator g;
    const int w = 64;
    const int bucket = BK_NEAR;
    int *pubIds = malloc(sizeof(int) * (size_t)w);
    int *refIds = malloc(sizeof(int) * (size_t)w);
    int64_t *refNp = malloc(sizeof(int64_t) * (size_t)w * NP_CH);
    if (!pubIds || !refIds || !refNp)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }

    const int modeBefore = gpuGetMode();
    gpuSetMode(GPU_MODE_AUTO);
    const uint64_t countBefore = gpuDispatchCount();
    const struct { uint64_t seed; uint32_t flags; } passes[] =
    {
        { seedA, flags },
        { seedB, flags },
        { seedA, flags | LARGE_BIOMES },
    };
    long long idDrift = 0;
    for (int p = 0; p < 3; p++)
    {
        setupGenerator(&g, mc, passes[p].flags);
        applySeed(&g, DIM_OVERWORLD, passes[p].seed);
        cpuSurfaceBiomeRef(refIds, refNp, &g.bn, g.sha, 4, 0, 0, w, 1);
        mapOverworldSurfaceBiome(pubIds, &g.bn, g.sha, 4, 0, 0, w, 1);
        for (int i = 0; i < w; i++)
        {
            accountIds(bucket, pubIds[i], refIds[i]);
            if (pubIds[i] != refIds[i])
                idDrift++;
        }
    }
    int shapeFail = 0;
#ifdef CUBIOMES_ENABLE_GPU
    if (gpuDispatchCount() - countBefore != 3)
    {
        fprintf(stderr, "gpu-test: reseed case engaged %llu/3 dispatches\n",
            (unsigned long long) (gpuDispatchCount() - countBefore));
        shapeFail = 1;
    }
#else
    (void) countBefore;
#endif
    gpuSetMode(modeBefore);

    printf("  -> %s: id drift %lld\n", shapeFail ? "FAIL" : "ok", idDrift);
    fflush(stdout);
    free(pubIds);
    free(refIds);
    free(refNp);
    total->pts += 3 * w;
    if (shapeFail)
        *fail = 1;
}

//----------------------------------------------------------------------
// Terrain overview heights: the guarded dispatch (gpuTryHeightHook) and the
// transparent mapOverworldTerrainHeight()/mapApproxHeight() entries, against
// the pure-CPU reference (cpuHeightRef pins the engine off). Heights and
// surface ids are budgeted; shape invariance and off-mode equality stay
// bit-exact.

typedef struct
{
    int stride;     // 1 = one dense 2D call, > 1 = strided per-row calls
    int w, h;
    int ids;        // also compare the biome ids at the surface
} HShape;

static const HShape HSHAPES[] =
{
    {1, DENSE_W, DENSE_H, 1},                   // dense quart map with ids
    {STRIDED_STRIDE, STRIDED_W, STRIDED_H, 1},  // strided overview rows
    {1, 4, 4, 1},                               // 4x4 chunk tile
    {1, DENSE_W, DENSE_H, 0},                   // dense quart map without ids
};
#define HSHAPE_CNT (int)(sizeof(HSHAPES) / sizeof(HSHAPES[0]))

static const char *hshapeNameByIdx(int idx)
{
    static const char *names[] = { "dense-ids", "strided-ids", "tile4x4-ids",
        "dense-noids" };
    return names[idx];
}

/*
 * One (version, seed, origin, shape) height case: the guarded GPU dispatch
 * vs the CPU reference. Dense shapes go through a single 2D call per side
 * (the tile/map usage), strided shapes through per-row calls (the mod's
 * overview-row streaming).
 */
static void runHeightCase(int mc, uint32_t flags, uint64_t seed, int ox,
    int oz, int shapeIdx, CaseStats *total, int *fail)
{
    const HShape *sh = &HSHAPES[shapeIdx];
    const int bucket = bucketOf(ox, oz);
    printf("height case mc=%d seed=%llu origin=(%d,%d) %s [%s]\n",
        mc, (unsigned long long) seed, ox, oz, hshapeNameByIdx(shapeIdx),
        bucketNames[bucket]);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);

    const int stride = sh->stride;
    const int rawW = (sh->w - 1) * stride + 1;
    float *refY = malloc(sizeof(float) * (size_t)rawW * sh->h);
    float *gpuY = malloc(sizeof(float) * (size_t)rawW * sh->h);
    int *refIds = sh->ids ? malloc(sizeof(int) * (size_t)rawW * sh->h) : NULL;
    int *gpuIds = sh->ids ? malloc(sizeof(int) * (size_t)rawW * sh->h) : NULL;
    if (!refY || !gpuY || (sh->ids && (!refIds || !gpuIds)))
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }

    long long pts = 0, idDrift = 0, heightDrift = 0;
    double tRef = 0.0, tGpu = 0.0;
    const int calls = stride == 1 ? 1 : sh->h;
    for (int c = 0; c < calls; c++)
    {
        const int rowZ = oz + c * stride;
        const int rw = stride == 1 ? sh->w : rawW;
        const int rh = stride == 1 ? sh->h : 1;
        tRef += cpuHeightRef(refY, refIds, &g.bn, ox, rowZ, rw, rh);
        double t0 = nowSeconds();
        if (gpuTryHeightHook(gpuY, gpuIds, &g.bn, ox, rowZ, rw, rh))
        {
            fprintf(stderr, "gpu-test: GPU height dispatch failed at mc=%d "
                "seed=%llu (%d,%d) - engine reported unavailable\n",
                mc, (unsigned long long) seed, ox, rowZ);
            exit(2);
        }
        double t1 = nowSeconds();
        tGpu += t1 - t0;

        for (int j = 0; j < rh; j++)
        {
            for (int i = 0; i < rw; i += stride)
            {
                const int idx = j * rw + i;
                pts++;
                accountHeight(bucket, gpuY[idx], refY[idx]);
                if (gpuY[idx] != refY[idx])
                {
                    heightDrift++;
                    if (heightDrift <= 3)
                        printf("  height @(%d,%d): gpu=%f cpu=%f\n",
                            ox + i, rowZ + j, (double) gpuY[idx],
                            (double) refY[idx]);
                }
                if (sh->ids)
                {
                    accountIds(bucket, gpuIds[idx], refIds[idx]);
                    if (gpuIds[idx] != refIds[idx])
                    {
                        idDrift++;
                        if (idDrift <= 3)
                            printf("  id @(%d,%d): gpu=%d cpu=%d\n",
                                ox + i, rowZ + j, gpuIds[idx], refIds[idx]);
                    }
                }
            }
        }
    }

    printf("  -> %s: %lld pts, height drift %lld, id drift %lld "
        "[cpu %.1fs gpu %.1fs]\n",
        heightDrift || idDrift ? "drift" : "ok",
        pts, heightDrift, idDrift, tRef, tGpu);
    fflush(stdout);
    free(refY);
    free(gpuY);
    free(refIds);
    free(gpuIds);
    total->pts += pts;
}

/*
 * Transparent height dispatch through the public mapOverworldTerrainHeight()
 * entry in off/auto/force: the off pass runs the CPU body and must match the
 * pure-CPU reference bit-exactly; auto/force run the GPU and are budgeted.
 * auto/force must engage the engine on every public call
 * (gpuHeightDispatchCount, CUBIOMES_ENABLE_GPU builds), off must not touch
 * it. The auto pass also re-queries sampled points as isolated single
 * columns: one strided/dense row dispatch must equal the isolated calls
 * bit-exactly (GPU-vs-GPU determinism).
 */
static void runTransparentHeightCase(int mc, uint32_t flags, uint64_t seed,
    int ox, int oz, CaseStats *total, int *fail)
{
    const int w = 64;
    const int rows = 2;
    const int bucket = bucketOf(ox, oz);
    printf("transparent height case mc=%d seed=%llu origin=(%d,%d) "
        "w=%d rows=%d + isolated [%s]\n",
        mc, (unsigned long long) seed, ox, oz, w, rows,
        bucketNames[bucket]);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);

    float *refY = malloc(sizeof(float) * (size_t)w * rows);
    int *refIds = malloc(sizeof(int) * (size_t)w * rows);
    float *pubY = malloc(sizeof(float) * (size_t)w * rows);
    int *pubIds = malloc(sizeof(int) * (size_t)w * rows);
    if (!refY || !refIds || !pubY || !pubIds)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }

    static const int modes[] =
        { GPU_MODE_OFF, GPU_MODE_AUTO, GPU_MODE_FORCE };
    static const char *modeNames[] = { "off", "auto", "force" };
    const int modeBefore = gpuGetMode();
    long long heightHard = 0, idHard = 0, heightDrift = 0, idDrift = 0;
    int shapeFail = 0;

    for (int p = 0; p < 3; p++)
    {
        for (int j = 0; j < rows; j++)
            cpuHeightRef(refY + (size_t)j * w, refIds + (size_t)j * w,
                &g.bn, ox, oz + j, w, 1);

        gpuSetMode(modes[p]);
        uint64_t before = gpuHeightDispatchCount();
        for (int j = 0; j < rows; j++)
            mapOverworldTerrainHeight(pubY + (size_t)j * w,
                pubIds + (size_t)j * w, &g.bn, ox, oz + j, w, 1);
        uint64_t engaged = gpuHeightDispatchCount() - before;
        uint64_t expected = 0;
#ifdef CUBIOMES_ENABLE_GPU
        if (modes[p] != GPU_MODE_OFF)
            expected = (uint64_t) rows;
        if (engaged != expected)
        {
            fprintf(stderr, "gpu-test: public height entry %s mode engaged "
                "%llu/%llu dispatches (mc=%d seed=%llu)\n", modeNames[p],
                (unsigned long long) engaged, (unsigned long long) expected,
                mc, (unsigned long long) seed);
            shapeFail = 1;
        }
#endif
        for (int j = 0; j < rows; j++)
        {
            for (int i = 0; i < w; i++)
            {
                const int idx = j * w + i;
                const int yDiff = pubY[idx] != refY[idx];
                const int idDiff = pubIds[idx] != refIds[idx];
                if (modes[p] == GPU_MODE_OFF)
                {
                    // CPU body: regression, never drift
                    if (yDiff)
                    {
                        heightHard++;
                        if (heightHard <= 3)
                            printf("  %s height @(%d,%d): pub=%f ref=%f\n",
                                modeNames[p], ox + i, oz + j,
                                (double) pubY[idx], (double) refY[idx]);
                    }
                    if (idDiff)
                    {
                        idHard++;
                        if (idHard <= 3)
                            printf("  %s id @(%d,%d): pub=%d ref=%d\n",
                                modeNames[p], ox + i, oz + j,
                                pubIds[idx], refIds[idx]);
                    }
                }
                else
                {
                    accountHeight(bucket, pubY[idx], refY[idx]);
                    accountIds(bucket, pubIds[idx], refIds[idx]);
                    heightDrift += yDiff;
                    idDrift += idDiff;
                }
            }
        }
        printf("  -> %s: hard mismatch %lld, height drift %lld, id drift %lld"
            " [gpu %llu/%llu]\n", modeNames[p], heightHard + idHard,
            heightDrift, idDrift,
            (unsigned long long) (modes[p] != GPU_MODE_OFF ? engaged : 0),
            (unsigned long long) (modes[p] != GPU_MODE_OFF ? expected : 0));
        fflush(stdout);
    }

    // Shape invariance: isolated single-column queries through the same
    // public entry must reproduce the row dispatch's heights and ids
    // bit-exactly (GPU-vs-GPU determinism, not drift).
    gpuSetMode(GPU_MODE_AUTO);
    for (int i = 0; i < w; i += 7)
    {
        float oneY;
        int oneId;
        mapOverworldTerrainHeight(&oneY, &oneId, &g.bn, ox + i, oz, 1, 1);
        if (oneY != pubY[i] || oneId != pubIds[i])
        {
            heightHard++;
            if (heightHard <= 3)
                printf("  isolated @(%d,%d): one=(%f,%d) row=(%f,%d)\n",
                    ox + i, oz, (double) oneY, oneId,
                    (double) pubY[i], pubIds[i]);
        }
    }
    gpuSetMode(modeBefore);

    free(refY);
    free(refIds);
    free(pubY);
    free(pubIds);
    total->pts += (long long) w * rows;
    if (heightHard || idHard || shapeFail)
        *fail = 1;
}

/*
 * The Generator-facing consumer entry mapApproxHeight(): on 1.18+ Overworld
 * it gates nptype and delegates to mapOverworldTerrainHeight(). The tile and
 * single-column shapes real consumers use, in auto mode, against the
 * mode-off CPU reference (budgeted).
 */
static void runApproxHeightCase(int mc, uint32_t flags, uint64_t seed,
    int ox, int oz, CaseStats *total, int *fail)
{
    const int bucket = bucketOf(ox, oz);
    printf("transparent mapApproxHeight case mc=%d seed=%llu origin=(%d,%d) "
        "tile4x4 + single [%s]\n",
        mc, (unsigned long long) seed, ox, oz, bucketNames[bucket]);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);
    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_OVERWORLD, g.seed);

    const int w = 4, h = 4;
    float refY[16], pubY[16];
    int refIds[16], pubIds[16];
    long long heightDrift = 0, idDrift = 0;

    cpuHeightRef(refY, refIds, &g.bn, ox, oz, w, h);
    mapApproxHeight(pubY, pubIds, &g, &sn, ox, oz, w, h);
    for (int i = 0; i < w * h; i++)
    {
        accountHeight(bucket, pubY[i], refY[i]);
        accountIds(bucket, pubIds[i], refIds[i]);
        if (pubY[i] != refY[i])
            heightDrift++;
        if (pubIds[i] != refIds[i])
            idDrift++;
    }

    float oneRefY, onePubY;
    int oneRefId, onePubId;
    cpuHeightRef(&oneRefY, &oneRefId, &g.bn, ox + 2, oz + 2, 1, 1);
    mapApproxHeight(&onePubY, &onePubId, &g, &sn, ox + 2, oz + 2, 1, 1);
    accountHeight(bucket, onePubY, oneRefY);
    accountIds(bucket, onePubId, oneRefId);
    if (onePubY != oneRefY)
        heightDrift++;
    if (onePubId != oneRefId)
        idDrift++;

    printf("  -> %s: height drift %lld, id drift %lld\n",
        heightDrift || idDrift ? "drift" : "ok", heightDrift, idDrift);
    fflush(stdout);
    total->pts += (long long) w * h + 1;
}

int main(int argc, char **argv)
{
    int full = 0;
    int reportOnly = 0;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--full"))
            full = 1;
        else if (!strcmp(argv[i], "--report"))
            reportOnly = 1;
        else
        {
            fprintf(stderr, "usage: gpu_test [--full] [--report]\n");
            return 2;
        }
    }

    Generator probe;
    setupGenerator(&probe, MC_1_21_4, 0);
    applySeed(&probe, DIM_OVERWORLD, 1ULL);
    if (gpuGetMode() == GPU_MODE_OFF)
    {
        // OFF must leave the runtime untouched even through real dispatch
        // calls (deterministic accessor, not a timing probe).
        float y[1];
        int ids[1];
        if (gpuRuntimeInitialized())
        {
            printf("FAIL: runtime initialized before any dispatch in off"
                " mode\n");
            return 1;
        }
        mapOverworldSurfaceBiome(ids, &probe.bn, probe.sha, 4, 0, 0, 1, 1);
        gpuTryHeightHook(y, ids, &probe.bn, 0, 0, 1, 1);
        if (gpuRuntimeInitialized())
        {
            printf("FAIL: runtime initialized in off mode\n");
            return 1;
        }
        printf("skip: CUBIOMES_GPU=off (runtime untouched, verified)\n");
        return 0;
    }
    if (!gpuAvailable(&probe))
    {
        printf("skip: no usable OpenCL device (CUBIOMES_GPU=%s)\n",
            gpuGetMode() == GPU_MODE_FORCE ? "force" : "auto");
        return 0;
    }
    printf("OpenCL device: %s\n", gpuDeviceName());
    printf("drift matrix: %s (CPU double reference vs GPU float; determinism"
        " contracts bit-exact, drift per budget)\n\n",
        full ? "FULL" : "default (--full extends)");
    fflush(stdout);

    static const uint64_t seeds[] =
    {
        146008555ULL,
        12345678901234567ULL,
        42ULL,
    };
    static const int origins[][2] =
    {
        {      0,       0},
        { 100000,  100000},
        {1000000, 1000000},
        {7500000, 7500000}, // quart = 30M blocks (world-border extreme)
        {-100000, -100000}, // negative quadrant, far field
        {   -48,     -48}, // 96-wide shapes straddle the zero coordinate
        {-7500000, -7500000}, // negative world-border extreme
    };

    static const int versions[] =
    {
        MC_1_18, MC_1_19_2, MC_1_19_4, MC_1_20_6,
        MC_1_21_4, MC_1_21_5, MC_26_2, MC_26_3,
    };
    const int mainMc = MC_1_21_4;

    int fail = 0;
    CaseStats total = {0};

    // full matrix on the main version
    const int seedCnt = full ? (int)(sizeof(seeds) / sizeof(seeds[0])) : 1;
    for (int si = 0; si < seedCnt; si++)
    {
        for (int oi = 0; oi < (int)(sizeof(origins) / sizeof(origins[0])); oi++)
        {
            for (int shape = 0; shape < SHAPE_CNT; shape++)
                runAndAccount(mainMc, 0, seeds[si], origins[oi][0],
                    origins[oi][1], shape, &total, &fail);
        }
    }

    // every other biome-tree version: reduced origin set (full also widens
    // it to the world border), one shape by default
    for (size_t vi = 0; vi < sizeof(versions) / sizeof(versions[0]); vi++)
    {
        if (versions[vi] == mainMc)
            continue;
        for (int oi = 0; oi < (int)(sizeof(origins) / sizeof(origins[0])); oi++)
        {
            // full sweeps both world-border extremes on every version
            if (origins[oi][0] != 0 && !(full && (origins[oi][0] == 7500000
                    || origins[oi][0] == -7500000)))
                continue;
            const int shapeCnt = full ? SHAPE_CNT : 1;
            for (int shape = 0; shape < shapeCnt; shape++)
                runAndAccount(versions[vi], 0, seeds[0], origins[oi][0],
                    origins[oi][1], shape, &total, &fail);
        }
    }

    // large biomes variant (different climate octave setup)
    for (int shape = 0; shape < 2; shape++)
        runAndAccount(mainMc, LARGE_BIOMES, seeds[0], origins[0][0],
            origins[0][1], shape, &total, &fail);

    // Transparent dispatch path through the public mapOverworldSurfaceBiome
    // entry (guarded hook). Default: the main version over the first two
    // origins and both dense shapes plus one row-vs-isolated case; --full
    // extends to all origins/shapes and every supported biome-tree version.
#ifndef CUBIOMES_ENABLE_GPU
    printf("transparent path: CUBIOMES_ENABLE_GPU not compiled in, "
        "public-entry cases run the CPU body (engagement checks off)\n\n");
#endif
    const int tshapeCnt = full
        ? (int)(sizeof(TSHAPES) / sizeof(TSHAPES[0]))
        : 2;
    for (int oi = 0; oi < (int)(sizeof(origins) / sizeof(origins[0])); oi++)
    {
        if (!full && oi > 1)
            continue;
        for (int shape = 0; shape < tshapeCnt; shape++)
            runTransparentCase(mainMc, 0, seeds[0], origins[oi][0],
                origins[oi][1], shape, &total, &fail);
        runTransparentShapeCase(mainMc, 0, seeds[0], origins[oi][0],
            origins[oi][1], &total, &fail);
    }
    for (size_t vi = 0; vi < sizeof(versions) / sizeof(versions[0]); vi++)
    {
        if (versions[vi] == mainMc)
            continue;
        if (!full && versions[vi] != MC_1_18 && versions[vi] != MC_26_3)
            continue;
        runTransparentCase(versions[vi], 0, seeds[0], 0, 0, 0, &total, &fail);
    }
    runTransparentReseedCase(mainMc, 0, seeds[0], seeds[1], &total, &fail);

    // Terrain overview heights: guarded dispatch matrix over the consumer
    // query shapes, then the transparent public entries. Default: main
    // version, one seed, all origins, the ids shapes plus the version sweep
    // at the origin; --full extends to all seeds and shapes and sweeps the
    // world-border origin on the other versions.
    const int hshapeCnt = full ? HSHAPE_CNT : 3;
    for (int si = 0; si < seedCnt; si++)
    {
        for (int oi = 0; oi < (int)(sizeof(origins) / sizeof(origins[0])); oi++)
        {
            for (int shape = 0; shape < hshapeCnt; shape++)
                runHeightCase(mainMc, 0, seeds[si], origins[oi][0],
                    origins[oi][1], shape, &total, &fail);
        }
    }
    for (size_t vi = 0; vi < sizeof(versions) / sizeof(versions[0]); vi++)
    {
        if (versions[vi] == mainMc)
            continue;
        for (int oi = 0; oi < (int)(sizeof(origins) / sizeof(origins[0])); oi++)
        {
            // full sweeps both world-border extremes on every version
            if (origins[oi][0] != 0 && !(full && (origins[oi][0] == 7500000
                    || origins[oi][0] == -7500000)))
                continue;
            runHeightCase(versions[vi], 0, seeds[0], origins[oi][0],
                origins[oi][1], 0, &total, &fail);
        }
    }
    runHeightCase(mainMc, LARGE_BIOMES, seeds[0], origins[0][0],
        origins[0][1], 0, &total, &fail);
    for (int oi = 0; oi < (int)(sizeof(origins) / sizeof(origins[0])); oi++)
    {
        if (!full && oi > 1)
            continue;
        runTransparentHeightCase(mainMc, 0, seeds[0], origins[oi][0],
            origins[oi][1], &total, &fail);
    }
    runTransparentHeightCase(MC_1_18, 0, seeds[0], 0, 0, &total, &fail);
    runTransparentHeightCase(MC_26_3, 0, seeds[0], 0, 0, &total, &fail);
    runApproxHeightCase(mainMc, 0, seeds[0], origins[0][0], origins[0][1],
        &total, &fail);

    printf("\n");
    if (total.wrapperReplicationMismatch)
    {
        printf("HARNESS BROKEN: reference replication mismatches the "
            "wrapper: %lld\n", total.wrapperReplicationMismatch);
        return 2;
    }
    if (fail)
    {
        printf("CONTRACT FAIL: %lld points compared; determinism contract "
            "violated (see FAIL lines above)\n", total.pts);
        return 1;
    }
    if (reportBudgets(!reportOnly))
    {
        printf("BUDGET FAIL: drift exceeded the accepted budgets\n");
        return 1;
    }
    printf("BUDGET OK: %lld points%s\n", total.pts,
        reportOnly ? " (report mode, budgets not asserted)" : "");
    return 0;
}
