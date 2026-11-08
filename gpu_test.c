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
 */

/*
 * z-stride row batching: the batched entries fold a whole strided row set
 * into ONE dispatch. Each case gates two things: batched vs the CPU
 * reference (budgeted through the same three-bucket accounting) and batched
 * vs the row-by-row guarded dispatch (bit-identical - same kernel, same
 * per-point math, only a different NDRange). Strides 16 (the mod's overview
 * rows) plus 7 and 33 catch z-decode bugs; negative origins ride along.
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

//----------------------------------------------------------------------
// z-stride row batching: the batched entries
// (gpuMapOverworldSurfaceBiomeRows / gpuMapOverworldTerrainHeightRows)
// serve a whole strided row set in ONE dispatch. Per case: batched vs the
// CPU reference (budgeted, three-bucket accounting) and batched vs the
// row-by-row guarded dispatch (bit-identical: same kernel, same per-point
// math, only a different NDRange). Degenerate shapes must reject without a
// dispatch.

static void runZStrideCase(int mc, uint32_t flags, uint64_t seed, int ox,
    int oz, int w, int rows, int zStride, int scale, CaseStats *total,
    int *fail)
{
    const int bucket = bucketOf(ox, oz);
    printf("z-stride case mc=%d seed=%llu origin=(%d,%d) w=%d rows=%d "
        "zStride=%d scale=%d [%s]\n",
        mc, (unsigned long long) seed, ox, oz, w, rows, zStride, scale,
        bucketNames[bucket]);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);

    const size_t pts = (size_t) w * rows;
    int *refIds = malloc(sizeof(int) * pts);
    int *rowIds = malloc(sizeof(int) * pts);
    int *batchIds = malloc(sizeof(int) * pts);
    int64_t *refNp = malloc(sizeof(int64_t) * pts * NP_CH);
    int64_t *rowNp = malloc(sizeof(int64_t) * pts * NP_CH);
    float *refY = malloc(sizeof(float) * pts);
    float *rowY = malloc(sizeof(float) * pts);
    float *batchY = malloc(sizeof(float) * pts);
    int *refHids = malloc(sizeof(int) * pts);
    int *rowHids = malloc(sizeof(int) * pts);
    int *batchHids = malloc(sizeof(int) * pts);
    if (!refIds || !rowIds || !batchIds || !refNp || !rowNp
        || !refY || !rowY || !batchY || !refHids || !rowHids || !batchHids)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }

    // Pure-CPU references, one row at a time (both batched entries take the
    // same per-row coordinates).
    for (int j = 0; j < rows; j++)
    {
        cpuSurfaceBiomeRef(refIds + (size_t) j * w,
            refNp + (size_t) j * w * NP_CH, &g.bn, g.sha, scale,
            ox, oz + j * zStride, w, 1);
        cpuHeightRef(refY + (size_t) j * w, refHids + (size_t) j * w,
            &g.bn, ox, oz + j * zStride, w, 1);
    }

    const int modeBefore = gpuGetMode();
    gpuSetMode(GPU_MODE_AUTO);

    uint64_t bio0 = gpuDispatchCount();
    for (int j = 0; j < rows; j++)
        gpuMapOverworldSurfaceBiome(rowIds + (size_t) j * w, &g, g.sha,
            scale, ox, oz + j * zStride, w, 1,
            rowNp + (size_t) j * w * NP_CH);
    uint64_t bioRows = gpuDispatchCount() - bio0;
    uint64_t bio1 = gpuDispatchCount();
    if (gpuMapOverworldSurfaceBiomeRows(batchIds, &g.bn, g.sha, scale,
            ox, oz, w, rows, zStride))
    {
        fprintf(stderr, "gpu-test: batched biome dispatch failed at mc=%d "
            "seed=%llu (%d,%d) w=%d rows=%d zStride=%d\n", mc,
            (unsigned long long) seed, ox, oz, w, rows, zStride);
        exit(2);
    }
    uint64_t bioBatch = gpuDispatchCount() - bio1;

    uint64_t h0 = gpuHeightDispatchCount();
    for (int j = 0; j < rows; j++)
        gpuTryHeightHook(rowY + (size_t) j * w, rowHids + (size_t) j * w,
            &g.bn, ox, oz + j * zStride, w, 1);
    uint64_t hRows = gpuHeightDispatchCount() - h0;
    uint64_t h1 = gpuHeightDispatchCount();
    if (gpuMapOverworldTerrainHeightRows(batchY, batchHids, &g.bn, ox, oz,
            w, rows, zStride))
    {
        fprintf(stderr, "gpu-test: batched height dispatch failed at mc=%d "
            "seed=%llu (%d,%d) w=%d rows=%d zStride=%d\n", mc,
            (unsigned long long) seed, ox, oz, w, rows, zStride);
        exit(2);
    }
    uint64_t hBatch = gpuHeightDispatchCount() - h1;
    gpuSetMode(modeBefore);

    long long idDrift = 0, npDrift = 0, heightDrift = 0, hidDrift = 0;
    long long biomeBit = 0, heightBit = 0;
    for (int j = 0; j < rows; j++)
    {
        for (int i = 0; i < w; i++)
        {
            const size_t idx = (size_t) j * w + i;
            // batched vs CPU reference: budgeted drift
            accountIds(bucket, batchIds[idx], refIds[idx]);
            if (batchIds[idx] != refIds[idx])
            {
                idDrift++;
                if (idDrift <= 3)
                    printf("  biome id @(%d,%d): batch=%d cpu=%d\n",
                        ox + i, oz + j * zStride, batchIds[idx],
                        refIds[idx]);
            }
            accountNp(bucket, rowNp + idx * NP_CH, refNp + idx * NP_CH);
            for (int k = 0; k < NP_CH; k++)
                if (rowNp[idx * NP_CH + k] != refNp[idx * NP_CH + k])
                    npDrift++;
            accountHeight(bucket, batchY[idx], refY[idx]);
            if (batchY[idx] != refY[idx])
                heightDrift++;
            accountIds(bucket, batchHids[idx], refHids[idx]);
            if (batchHids[idx] != refHids[idx])
                hidDrift++;
            // batched vs row-by-row GPU: bit-identical, never drift
            if (batchIds[idx] != rowIds[idx])
            {
                biomeBit++;
                if (biomeBit <= 3)
                    printf("  biome bit @(%d,%d): batch=%d rows=%d\n",
                        ox + i, oz + j * zStride, batchIds[idx],
                        rowIds[idx]);
            }
            if (batchY[idx] != rowY[idx] || batchHids[idx] != rowHids[idx])
            {
                heightBit++;
                if (heightBit <= 3)
                    printf("  height bit @(%d,%d): batch=(%f,%d) "
                        "rows=(%f,%d)\n",
                        ox + i, oz + j * zStride, (double) batchY[idx],
                        batchHids[idx], (double) rowY[idx], rowHids[idx]);
            }
        }
    }

    // Engagement: one dispatch per entry form (batched = 1, rows = rows).
    int shapeFail = 0;
#ifdef CUBIOMES_ENABLE_GPU
    if (bioBatch != 1 || bioRows != (uint64_t) rows
        || hBatch != 1 || hRows != (uint64_t) rows)
    {
        fprintf(stderr, "gpu-test: z-stride engagement biome %llu(batch)/"
            "%llu(rows) height %llu(batch)/%llu(rows), expected 1/%d each"
            " (mc=%d)\n",
            (unsigned long long) bioBatch, (unsigned long long) bioRows,
            (unsigned long long) hBatch, (unsigned long long) hRows,
            rows, mc);
        shapeFail = 1;
    }
#else
    (void) bioBatch; (void) bioRows; (void) hBatch; (void) hRows;
#endif

    // Argument guard: degenerate shapes reject before the engine is touched
    // (nonzero return, zero new dispatches, in any mode).
    int junk[2];
    float junkY[2];
    gpuSetMode(GPU_MODE_AUTO);
    uint64_t g0 = gpuDispatchCount() + gpuHeightDispatchCount();
    int r1 = gpuMapOverworldSurfaceBiomeRows(junk, &g.bn, g.sha, 4,
        ox, oz, 2, 2, 0);
    int r2 = gpuMapOverworldSurfaceBiomeRows(junk, &g.bn, g.sha, 4,
        ox, oz, 0, 2, zStride);
    int r3 = gpuMapOverworldSurfaceBiomeRows(junk, &g.bn, g.sha, 4,
        ox, oz, 2, (1 << 20), zStride); // over the dispatch cap
    int r4 = gpuMapOverworldTerrainHeightRows(junkY, junk, &g.bn,
        ox, oz, 2, -1, zStride);
    uint64_t g1 = gpuDispatchCount() + gpuHeightDispatchCount();
    gpuSetMode(modeBefore);
    if (!(r1 && r2 && r3 && r4) || g1 != g0)
    {
        fprintf(stderr, "gpu-test: z-stride argument guard leak "
            "(r=%d%d%d%d, dispatches %llu -> %llu)\n", r1, r2, r3, r4,
            (unsigned long long) g0, (unsigned long long) g1);
        shapeFail = 1;
    }

    printf("  -> %s: id drift %lld, np drift %lld, height drift %lld, "
        "height id drift %lld, bit-identical %s/%s [disp b %llu/%llu "
        "h %llu/%llu]\n",
        shapeFail || biomeBit || heightBit ? "FAIL"
            : (idDrift || npDrift || heightDrift || hidDrift) ? "drift"
            : "ok",
        idDrift, npDrift, heightDrift, hidDrift,
        biomeBit ? "FAIL" : "ok", heightBit ? "FAIL" : "ok",
        (unsigned long long) bioBatch, (unsigned long long) bioRows,
        (unsigned long long) hBatch, (unsigned long long) hRows);
    fflush(stdout);
    free(refIds);
    free(rowIds);
    free(batchIds);
    free(refNp);
    free(rowNp);
    free(refY);
    free(rowY);
    free(batchY);
    free(refHids);
    free(rowHids);
    free(batchHids);
    total->pts += (long long) pts;
    if (shapeFail || biomeBit || heightBit)
        *fail = 1;
}

//----------------------------------------------------------------------
// cfx product parity: the mod's shim hot path (confluxnative.c), bit-exact.
//
// confluxnative.c is JNI-bound, so the harness mirrors its analytic overview
// solver and caller-Y biome loop below. The shim compiles with the same
// -ffp-contract=off -fno-fast-math discipline as this file, and the
// gpu-cfxmap integration report's JNI-stub harness pins this replication
// against the real shim source. The biome stage needs no replication: the
// shim calls cubiomes' own voronoiAccess3D() + sampleBiomeNoise() directly,
// so the harness calls them too and the kernel must match bit-for-bit.

#define CFX_JAGGED_KEEP_OCTAVES 4
#define CFX_OVERVIEW_ITERATIONS 2

static double cfxMaintainPrecisionRef(double value)
{
    return value - floor(value / 33554432.0 + 0.5) * 33554432.0;
}

static float cfxPeaksAndValleysRef(float weirdness)
{
    return -(fabsf(fabsf(weirdness) - 0.6666667F) - 0.33333334F) * 3.0F;
}

static void cfxTerrainParametersRef(const BiomeNoise *bn, int cellX,
    int cellZ, float values[4])
{
    double px = cellX;
    double pz = cellZ;
    px += sampleDoublePerlin(&bn->climate[NP_SHIFT], cellX, 0, cellZ) * 4.0;
    pz += sampleDoublePerlin(&bn->climate[NP_SHIFT], cellZ, cellX, 0) * 4.0;
    const float weirdness = sampleDoublePerlin(
        &bn->climate[NP_WEIRDNESS], px, 0, pz);
    values[0] = sampleDoublePerlin(&bn->climate[NP_CONTINENTALNESS], px, 0, pz);
    values[1] = sampleDoublePerlin(&bn->climate[NP_EROSION], px, 0, pz);
    values[2] = cfxPeaksAndValleysRef(weirdness);
    values[3] = weirdness;
}

/* sampleDoublePerlin over only the first maxOct octaves (coarsest first). */
static double cfxSampleDoublePerlinPrefixRef(const DoublePerlinNoise *dp,
    double x, double y, double z, int maxOct)
{
    const double f = 337.0 / 331.0;
    double v = 0.0;
    int i;
    for (i = 0; i < maxOct && i < dp->octA.octcnt; i++)
    {
        const PerlinNoise *p = dp->octA.octaves + i;
        const double lf = p->lacunarity;
        v += p->amplitude * samplePerlin(p,
            maintainPrecision(x * lf), maintainPrecision(y * lf),
            maintainPrecision(z * lf), 0, 0);
    }
    for (i = 0; i < maxOct && i < dp->octB.octcnt; i++)
    {
        const PerlinNoise *p = dp->octB.octaves + i;
        const double lf = p->lacunarity;
        const double bx = x * f;
        const double by = y * f;
        const double bz = z * f;
        v += p->amplitude * samplePerlin(p,
            maintainPrecision(bx * lf), maintainPrecision(by * lf),
            maintainPrecision(bz * lf), 0, 0);
    }
    return v * dp->amplitude;
}

static double cfxModernTerrainNoiseRef(const SurfaceNoise *sn, int x, int y,
    int z, int mainFirst, int minMaxFirst)
{
    const double scaledXz = 684.412 * sn->xzScale;
    const double scaledY = 684.412 * sn->yScale;
    const double dx = x * scaledXz;
    const double dy = y * scaledY;
    const double dz = z * scaledXz;
    const double mainX = dx / sn->xzFactor;
    const double mainY = dy / sn->yFactor;
    const double mainZ = dz / sn->xzFactor;
    const double smearY = scaledY * 8.0;
    const double mainSmearY = smearY / sn->yFactor;
    double interpolation = 0.0;
    double frequency = ldexp(1.0, -mainFirst);
    int i;

    for (i = mainFirst; i < 8; i++)
    {
        interpolation += samplePerlin(&sn->octmain.octaves[i],
            cfxMaintainPrecisionRef(mainX * frequency),
            cfxMaintainPrecisionRef(mainY * frequency),
            cfxMaintainPrecisionRef(mainZ * frequency),
            mainSmearY * frequency,
            mainY * frequency) / frequency;
        frequency *= 0.5;
    }

    const double blend = (interpolation / 10.0 + 1.0) * 0.5;
    double lower = 0.0;
    double upper = 0.0;
    frequency = ldexp(1.0, -minMaxFirst);
    for (i = minMaxFirst; i < 16; i++)
    {
        const double sx = cfxMaintainPrecisionRef(dx * frequency);
        const double sy = cfxMaintainPrecisionRef(dy * frequency);
        const double sz = cfxMaintainPrecisionRef(dz * frequency);
        const double yScale = smearY * frequency;
        if (blend < 1.0)
            lower += samplePerlin(&sn->octmin.octaves[i],
                sx, sy, sz, yScale, dy * frequency) / frequency;
        if (blend > 0.0)
            upper += samplePerlin(&sn->octmax.octaves[i],
                sx, sy, sz, yScale, dy * frequency) / frequency;
        frequency *= 0.5;
    }
    return clampedLerp(blend, lower / 512.0, upper / 512.0) / 128.0;
}

static int cfxModernOverviewHeightRef(const BiomeNoise *bn, int x4, int z4,
    int mainFirst, int minMaxFirst, int truncated)
{
    float values[4];
    cfxTerrainParametersRef(bn, x4, z4, values);
    const double offset = -0.50375F + getSpline(bn->sp, values);
    const double factor = getSpline(bn->factorSp, values);
    const double jaggedness = getSpline(bn->jaggedSp, values);
    double jagged = 0.0;
    if (!truncated)
    {
        jagged = sampleDoublePerlin(&bn->jagged,
            x4 * 4 * 1500.0, 0.0, z4 * 4 * 1500.0);
        if (jagged < 0.0)
            jagged *= 0.5;
        jagged *= jaggedness;
    }
    else if (jaggedness != 0.0F)
    {
        jagged = cfxSampleDoublePerlinPrefixRef(&bn->jagged,
            x4 * 4 * 1500.0, 0.0, z4 * 4 * 1500.0, CFX_JAGGED_KEEP_OCTAVES);
        if (jagged < 0.0)
            jagged *= 0.5;
        jagged *= jaggedness;
    }

    const double base = 1.0 + offset + jagged;
    double y = 128.0 * base;
    if (factor <= 1e-9)
        return (int) floor(y);

    for (int i = 0; i < CFX_OVERVIEW_ITERATIONS; i++)
    {
        const double noise = cfxModernTerrainNoiseRef(&bn->terrain,
            x4 * 4, (int) floor(y), z4 * 4, mainFirst, minMaxFirst);
        const double above = 128.0 * (base + noise / factor);
        const double below = 128.0 * (base + noise / (4.0 * factor));
        const double next = above >= 128.0 * base ? above : below;
        if (fabs(next - y) < 0.5)
        {
            y = next;
            break;
        }
        y = next;
    }
    return (int) floor(y);
}

/* The shim's cfxNoiseDetailForStride(): coarse strides drop fine octaves. */
static void cfxNoiseDetailForStrideRef(const SurfaceNoise *sn, int stride,
    int *mainFirst, int *minMaxFirst, int *truncated)
{
    *mainFirst = 0;
    *minMaxFirst = 0;
    *truncated = 0;
    if (stride < 8)
        return;
    *truncated = 1;
    const double scaledXz = 684.412 * sn->xzScale;
    double w = 1.0 / scaledXz;
    while (*minMaxFirst < 16 && w < (double) stride * 0.5)
    {
        w *= 2.0;
        (*minMaxFirst)++;
    }
    w = 1.0 / (scaledXz / sn->xzFactor);
    while (*mainFirst < 8 && w < (double) stride * 0.25)
    {
        w *= 2.0;
        (*mainFirst)++;
    }
}

// Overview lattices the shim's tile pipeline solves: the dense LOD0-1
// quart macro grids (step 1, full detail) and the LOD2-4 coarse anchor
// lattices (stride/4 steps, truncated detail). The kernel parity runs on
// the lattice itself; the shim's integer interpolation between anchors is
// host-side code the GPU never touches.
typedef struct
{
    int step4;
    int stride;
    int w, h;
} CfxLShape;

static const CfxLShape CFX_LSHAPES[] =
{
    {1, 1, 66, 66},     // LOD0 dense overview macro lattice
    {1, 2, 130, 130},   // LOD1 macro lattice
    {2, 8, 130, 130},   // LOD2 coarse anchors
    {4, 16, 130, 130},  // LOD3 coarse anchors
    {8, 32, 66, 66},    // LOD4 coarse anchors (reduced size to bound runtime)
};
#define CFX_LSHAPE_CNT (int)(sizeof(CFX_LSHAPES) / sizeof(CFX_LSHAPES[0]))

// Caller-Y biome batches of the tile pipeline: dense LOD0-1 and the
// LOD2-4 anchor sweeps (the tiny lattice-refine 2x2 cells stay CPU by
// dispatch-overhead policy and are shape-covered by the small dense cases).
typedef struct
{
    int stride;
    int w, h;
} CfxBShape;

static const CfxBShape CFX_BSHAPES[] =
{
    {1, 258, 258},   // LOD0 dense
    {2, 129, 129},   // LOD1 dense
    {8, 130, 130},   // LOD2 anchors
    {16, 130, 130},  // LOD3 anchors
    {32, 130, 130},  // LOD4 anchors
};
#define CFX_BSHAPE_CNT (int)(sizeof(CFX_BSHAPES) / sizeof(CFX_BSHAPES[0]))

static int runCfxLatticeCase(const Generator *g, int ox4, int oz4,
    const CfxLShape *sh, CaseStats *total, int *fail)
{
    const int w = sh->w, h = sh->h, points = w * h;
    int mf, mmf, tr;
    cfxNoiseDetailForStrideRef(&g->bn.terrain, sh->stride, &mf, &mmf, &tr);

    int *ref = malloc(sizeof(int) * (size_t)points);
    int *gpu = malloc(sizeof(int) * (size_t)points);
    if (!ref || !gpu)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }
    double t0 = nowSeconds();
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            ref[j * w + i] = cfxModernOverviewHeightRef(&g->bn,
                ox4 + i * sh->step4, oz4 + j * sh->step4, mf, mmf, tr);
    double t1 = nowSeconds();
    int rc = gpuCfxOverviewHeights(gpu, &g->bn, ox4, oz4, w, h,
        sh->step4, sh->step4, mf, mmf, tr);
    double t2 = nowSeconds();
    if (rc != 0)
    {
        printf("  FAIL: gpuCfxOverviewHeights rejected (stride %d, rc %d)\n",
            sh->stride, rc);
        *fail = 1;
    }
    int mismatches = 0;
    for (int p = 0; p < points; p++)
        if (gpu[p] != ref[p])
            mismatches++;
    if (mismatches)
    {
        printf("  FAIL: overview lattice stride=%d: %d/%d height mismatches"
            " (mf=%d mmf=%d tr=%d)\n",
            sh->stride, mismatches, points, mf, mmf, tr);
        const int show = mismatches < 3 ? mismatches : 3;
        int shown = 0;
        for (int j = 0; j < h && shown < show; j++)
            for (int i = 0; i < w && shown < show; i++)
                if (gpu[j * w + i] != ref[j * w + i])
                {
                    printf("    @(x4=%d,z4=%d): gpu=%d ref=%d\n",
                        ox4 + i * sh->step4, oz4 + j * sh->step4,
                        gpu[j * w + i], ref[j * w + i]);
                    shown++;
                }
        *fail = 1;
    }
    printf("  overview lattice stride=%d %dx%d: %d mismatches"
        " [cpu %.0fms gpu %.0fms]\n",
        sh->stride, w, h, mismatches, (t1 - t0) * 1e3, (t2 - t1) * 1e3);
    total->pts += points;
    free(ref);
    free(gpu);
    return mismatches;
}

static int runCfxBiomeCase(const Generator *g, int ox, int oz,
    const CfxBShape *sh, CaseStats *total, int *fail)
{
    const int w = sh->w, h = sh->h, points = w * h;
    int mf, mmf, tr;
    cfxNoiseDetailForStrideRef(&g->bn.terrain, sh->stride, &mf, &mmf, &tr);

    int *terrainY = malloc(sizeof(int) * (size_t)points);
    int *ref = malloc(sizeof(int) * (size_t)points);
    int *gpu = malloc(sizeof(int) * (size_t)points);
    if (!terrainY || !ref || !gpu)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }
    // The caller-Y contract resolves biomes from already-sampled terrain;
    // feed the batch the same solver heights the product would.
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            terrainY[j * w + i] = cfxModernOverviewHeightRef(&g->bn,
                floordiv(ox + i * sh->stride, 4),
                floordiv(oz + j * sh->stride, 4), mf, mmf, tr);

    double t0 = nowSeconds();
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            const int idx = j * w + i;
            int x4, y4, z4;
            voronoiAccess3D(g->sha, ox + i * sh->stride,
                terrainY[idx] < 62 ? 62 : terrainY[idx],
                oz + j * sh->stride, &x4, &y4, &z4);
            ref[idx] = sampleBiomeNoise(&g->bn, NULL, x4, y4, z4, NULL, 0);
        }
    }
    double t1 = nowSeconds();
    int rc = gpuCfxSurfaceBiomes(gpu, &g->bn, g->sha, ox, oz, w, h,
        sh->stride, terrainY);
    double t2 = nowSeconds();
    if (rc != 0)
    {
        printf("  FAIL: gpuCfxSurfaceBiomes rejected (stride %d, rc %d)\n",
            sh->stride, rc);
        *fail = 1;
    }
    int mismatches = 0;
    for (int p = 0; p < points; p++)
        if (gpu[p] != ref[p])
            mismatches++;
    if (mismatches)
    {
        printf("  FAIL: caller-Y biomes stride=%d: %d/%d id mismatches\n",
            sh->stride, mismatches, points);
        const int show = mismatches < 3 ? mismatches : 3;
        int shown = 0;
        for (int j = 0; j < h && shown < show; j++)
            for (int i = 0; i < w && shown < show; i++)
                if (gpu[j * w + i] != ref[j * w + i])
                {
                    printf("    @(%d,%d): gpu=%d ref=%d\n",
                        ox + i * sh->stride, oz + j * sh->stride,
                        gpu[j * w + i], ref[j * w + i]);
                    shown++;
                }
        *fail = 1;
    }
    printf("  caller-Y biomes stride=%d %dx%d: %d mismatches"
        " [cpu %.0fms gpu %.0fms]\n",
        sh->stride, w, h, mismatches, (t1 - t0) * 1e3, (t2 - t1) * 1e3);
    total->pts += points;
    free(terrainY);
    free(ref);
    free(gpu);
    return mismatches;
}

static void runCfxCase(int mc, uint32_t flags, uint64_t seed, int ox, int oz,
    int shapeCnt, CaseStats *total, int *fail)
{
    printf("cfx parity mc=%d seed=%llu origin=(%d,%d)\n",
        mc, (unsigned long long) seed, ox, oz);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);
    if (g.bn.nptype >= 0)
    {
        fprintf(stderr, "gpu-test: generator for mc %d not initialized\n", mc);
        exit(2);
    }

    const int ox4 = floordiv(ox, 4);
    const int oz4 = floordiv(oz, 4);
    for (int s = 0; s < shapeCnt && s < CFX_LSHAPE_CNT; s++)
        runCfxLatticeCase(&g, ox4, oz4, &CFX_LSHAPES[s], total, fail);
    for (int s = 0; s < shapeCnt && s < CFX_BSHAPE_CNT; s++)
        runCfxBiomeCase(&g, ox, oz, &CFX_BSHAPES[s], total, fail);
}

// Mode handling of the cfx entries: off must decline without touching the
// outputs, force must still dispatch on a working device.
static void runCfxModeCase(uint64_t seed, CaseStats *total, int *fail)
{
    Generator g;
    setupGenerator(&g, MC_1_21_4, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    int ref[64], gpu[64], terrainY[64];
    for (int i = 0; i < 64; i++)
        terrainY[i] = 64;

    const int modeBefore = gpuGetMode();
    gpuSetMode(GPU_MODE_OFF);
    int rcH = gpuCfxOverviewHeights(gpu, &g.bn, 0, 0, 8, 8, 1, 1, 0, 0, 0);
    int rcB = gpuCfxSurfaceBiomes(gpu, &g.bn, g.sha, 0, 0, 8, 8, 1, terrainY);
    gpuSetMode(modeBefore);
    if (rcH == 0 || rcB == 0)
    {
        printf("  FAIL: cfx dispatch succeeded in off mode (%d %d)\n", rcH, rcB);
        *fail = 1;
    }

    gpuSetMode(GPU_MODE_FORCE);
    int rcF = gpuCfxOverviewHeights(gpu, &g.bn, 0, 0, 8, 8, 1, 1, 0, 0, 0);
    gpuSetMode(modeBefore);
    if (rcF != 0)
    {
        printf("  FAIL: cfx dispatch rejected in force mode (rc %d)\n", rcF);
        *fail = 1;
    }
    else
    {
        const int mf = 0, mmf = 0, tr = 0;
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++)
                ref[j * 8 + i] = cfxModernOverviewHeightRef(&g.bn,
                    i, j, mf, mmf, tr);
        if (memcmp(gpu, ref, sizeof(ref)) != 0)
        {
            printf("  FAIL: force-mode cfx heights drifted from the CPU\n");
            *fail = 1;
        }
        total->pts += 64;
    }
}

// Slab rotation regression: the engine rotates its output slabs per
// dispatch, so two back-to-back lattices land on different slabs. Both runs
// must match each other and the CPU reference bit-for-bit; cross-slab
// contamination flips this.
static void runCfxSlabRotationCase(uint64_t seed, CaseStats *total, int *fail)
{
    Generator g;
    setupGenerator(&g, MC_1_21_4, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    const CfxLShape *sh = &CFX_LSHAPES[1]; // the 130x130 LOD1 macro lattice
    const int w = sh->w, h = sh->h, points = w * h;
    int mf, mmf, tr;
    cfxNoiseDetailForStrideRef(&g.bn.terrain, sh->stride, &mf, &mmf, &tr);

    int *ref = malloc(sizeof(int) * (size_t)points);
    int *runA = malloc(sizeof(int) * (size_t)points);
    int *runB = malloc(sizeof(int) * (size_t)points);
    if (!ref || !runA || !runB)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            ref[j * w + i] = cfxModernOverviewHeightRef(&g.bn,
                i * sh->step4, j * sh->step4, mf, mmf, tr);

    int rcA = gpuCfxOverviewHeights(runA, &g.bn, 0, 0, w, h,
        sh->step4, sh->step4, mf, mmf, tr);
    int rcB = gpuCfxOverviewHeights(runB, &g.bn, 0, 0, w, h,
        sh->step4, sh->step4, mf, mmf, tr);
    if (rcA != 0 || rcB != 0)
    {
        printf("  FAIL: slab rotation dispatch refused (%d %d)\n", rcA, rcB);
        *fail = 1;
    }
    else if (memcmp(runA, runB, sizeof(int) * (size_t)points) != 0)
    {
        printf("  FAIL: slab rotation changed results between dispatches\n");
        *fail = 1;
    }
    else if (memcmp(runA, ref, sizeof(int) * (size_t)points) != 0)
    {
        printf("  FAIL: slab rotation drifted from the CPU reference\n");
        *fail = 1;
    }
    else
    {
        printf("  slab rotation: %dx%d twice back-to-back bit-identical\n",
            w, h);
        total->pts += points;
    }
    free(ref);
    free(runA);
    free(runB);
}

// The batch entry must fill every block's range of the caller's flat layout
// bit-exactly the way n per-block dispatches do. Mixed grid shapes, strides
// and origins make the per-block geometry (and kernel arg re-set between
// same-kernel enqueues) actually vary; the blocks pack the flat offsets
// with no gaps, derived from the shapes. Interleaving the reference calls
// with the batch call also walks the slab rotation across both entry kinds.
static void runCfxHeightsBatchCase(uint64_t seed, CaseStats *total, int *fail)
{
    Generator g;
    setupGenerator(&g, MC_1_21_4, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    enum { CNT = 3 };
    static const int w[CNT] = { 130, 65, 130 };
    static const int h[CNT] = { 130, 65, 65 };
    static const int step[CNT] = { 4, 8, 4 };
    static const int x4[CNT] = { 0, 500, -300 };
    static const int z4[CNT] = { 0, -200, 400 };
    // Full detail scalars for every block: the batch shares one set by
    // contract, and step/detail pairing policy belongs to the shim, not
    // this mechanism test.
    const int mf = 0, mmf = 0, tr = 0;

    long offs[CNT];
    int totalPts = 0;
    for (int i = 0; i < CNT; i++)
    {
        offs[i] = totalPts;
        totalPts += w[i] * h[i];
    }

    int *ref = malloc(sizeof(int) * (size_t)totalPts);
    int *batch = malloc(sizeof(int) * (size_t)totalPts);
    if (!ref || !batch)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }

    int rcRef = 0;
    for (int i = 0; i < CNT; i++)
        rcRef |= gpuCfxOverviewHeights(ref + offs[i], &g.bn, x4[i], z4[i],
            w[i], h[i], step[i], step[i], mf, mmf, tr);
    int failMask = (int) 0x5A5A5A5A;
    int rcBatch = gpuCfxOverviewHeightsBatch(&failMask, batch, offs, &g.bn,
        x4, z4, w, h, step, step, CNT, mf, mmf, tr);

    if (rcRef != 0 || rcBatch != 0)
    {
        printf("  FAIL: heights batch or per-block dispatch refused"
            " (%d %d)\n", rcRef, rcBatch);
        *fail = 1;
    }
    else if (failMask != 0)
    {
        printf("  FAIL: heights batch reported per-block failures"
            " (failMask %#x)\n", failMask);
        *fail = 1;
    }
    else
    {
        int mismatches = 0;
        for (int i = 0; i < CNT; i++)
        {
            const int pts = w[i] * h[i];
            const int *refBlk = ref + offs[i];
            const int *batchBlk = batch + offs[i];
            for (int p = 0; p < pts; p++)
                if (batchBlk[p] != refBlk[p])
                {
                    if (mismatches < 3)
                        printf("    block %d @%d: batch=%d ref=%d\n",
                            i, p, batchBlk[p], refBlk[p]);
                    mismatches++;
                }
        }
        if (mismatches)
        {
            printf("  FAIL: heights batch drifted from per-block dispatches:"
                " %d mismatches\n", mismatches);
            *fail = 1;
        }
        else
        {
            printf("  heights batch: %d blocks (%d pts) bit-identical to"
                " per-block dispatches\n", CNT, totalPts);
            total->pts += totalPts;
        }
    }
    free(ref);
    free(batch);
}

// Whole-batch refusal must fire before any lock, enqueue or output write:
// the first block is exactly the dispatch cap and a second block pushes the
// sum over it, while the heights target is deliberately tiny - any write
// before the cap check corrupts the stack here. failMask must stay
// untouched on refusal (the caller keys its fallback off the return value).
static void runCfxHeightsBatchRefusalCase(uint64_t seed, CaseStats *total,
    int *fail)
{
    Generator g;
    setupGenerator(&g, MC_1_21_4, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    // gpu.c's GPU_MAX_POINTS dispatch cap (kept private there).
    enum { CAP_POINTS = 1 << 20 };
    const int x4[2] = { 0, 16 };
    const int z4[2] = { 0, 16 };
    const int gw[2] = { CAP_POINTS, 1 };
    const int gh[2] = { 1, 1 };
    const int st[2] = { 4, 4 };
    const long offs[2] = { 0, 1 };

    int failMask = (int) 0x5A5A5A5A;
    int tiny = 0;
    int rc = gpuCfxOverviewHeightsBatch(&failMask, &tiny, offs, &g.bn, x4,
        z4, gw, gh, st, st, 2, 0, 0, 0);
    if (rc == 0)
    {
        printf("  FAIL: over-cap heights batch was attempted\n");
        *fail = 1;
    }
    else if (failMask != (int) 0x5A5A5A5A)
    {
        printf("  FAIL: over-cap refusal touched failMask (%#x)\n",
            failMask);
        *fail = 1;
    }
    else
    {
        printf("  heights batch refusal: cap+1 points rejected before any"
            " write\n");
    }
}

// The biome batch must fill every block's range of the caller's flat ids
// layout bit-exactly the way n per-block gpuCfxSurfaceBiomes() calls do,
// with the concatenated terrainY planes uploading as one packed write.
// Mixed shapes, strides and origins make the per-block geometry (and the
// kernel arg re-set between same-kernel enqueues) actually vary; comparing
// each block's output range independently also catches the flat terrainY
// write bleeding across blocks. The blocks pack the flat offsets with no
// gaps, derived from the shapes; interleaving the reference calls with the
// batch call walks the slab rotation across both entry kinds.
static void runCfxBiomesBatchCase(uint64_t seed, CaseStats *total, int *fail)
{
    Generator g;
    setupGenerator(&g, MC_1_21_4, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    enum { CNT = 3 };
    static const int w[CNT] = { 129, 130, 65 };
    static const int rows[CNT] = { 129, 65, 130 };
    static const int stride[CNT] = { 2, 8, 16 };
    static const int bx[CNT] = { 0, 800000, -300000 };
    static const int bz[CNT] = { 0, -400000, 500000 };

    long offs[CNT];
    int totalPts = 0;
    for (int i = 0; i < CNT; i++)
    {
        offs[i] = totalPts;
        totalPts += w[i] * rows[i];
    }

    int *terrainY = malloc(sizeof(int) * (size_t)totalPts);
    int *ref = malloc(sizeof(int) * (size_t)totalPts);
    int *batch = malloc(sizeof(int) * (size_t)totalPts);
    if (!terrainY || !ref || !batch)
    {
        fprintf(stderr, "gpu-test: out of memory\n");
        exit(2);
    }
    // One flat terrainY input shared by both entry kinds: block i's
    // row-major plane sits at offs[i], solver heights exactly like the
    // product feeds them (detail scalars are per stride).
    for (int i = 0; i < CNT; i++)
    {
        int mf, mmf, tr;
        cfxNoiseDetailForStrideRef(&g.bn.terrain, stride[i], &mf, &mmf, &tr);
        for (int j = 0; j < rows[i]; j++)
            for (int c = 0; c < w[i]; c++)
                terrainY[offs[i] + j * w[i] + c] =
                    cfxModernOverviewHeightRef(&g.bn,
                        floordiv(bx[i] + c * stride[i], 4),
                        floordiv(bz[i] + j * stride[i], 4), mf, mmf, tr);
    }

    int rcRef = 0;
    for (int i = 0; i < CNT; i++)
        rcRef |= gpuCfxSurfaceBiomes(ref + offs[i], &g.bn, g.sha, bx[i],
            bz[i], w[i], rows[i], stride[i], terrainY + offs[i]);
    int failMask = (int) 0x5A5A5A5A;
    int rcBatch = gpuCfxSurfaceBiomesBatch(&failMask, batch, offs, &g.bn,
        g.sha, bx, bz, w, rows, stride, terrainY, CNT);

    if (rcRef != 0 || rcBatch != 0)
    {
        printf("  FAIL: biomes batch or per-block dispatch refused"
            " (%d %d)\n", rcRef, rcBatch);
        *fail = 1;
    }
    else if (failMask != 0)
    {
        printf("  FAIL: biomes batch reported per-block failures"
            " (failMask %#x)\n", failMask);
        *fail = 1;
    }
    else
    {
        int mismatches = 0;
        for (int i = 0; i < CNT; i++)
        {
            const int pts = w[i] * rows[i];
            const int *refBlk = ref + offs[i];
            const int *batchBlk = batch + offs[i];
            for (int p = 0; p < pts; p++)
                if (batchBlk[p] != refBlk[p])
                {
                    if (mismatches < 3)
                        printf("    block %d @%d: batch=%d ref=%d\n",
                            i, p, batchBlk[p], refBlk[p]);
                    mismatches++;
                }
        }
        if (mismatches)
        {
            printf("  FAIL: biomes batch drifted from per-block dispatches:"
                " %d mismatches\n", mismatches);
            *fail = 1;
        }
        else
        {
            printf("  biomes batch: %d blocks (%d pts) bit-identical to"
                " per-block dispatches\n", CNT, totalPts);
            total->pts += totalPts;
        }
    }
    free(terrainY);
    free(ref);
    free(batch);
}

// Whole-batch refusal must fire before any lock, upload, enqueue or output
// write: the first block is exactly the dispatch cap and a second block
// pushes the sum over it, while the ids target is deliberately one int -
// any premature write corrupts visibly here. failMask must stay untouched
// on refusal (the caller keys its fallback off the return value).
static void runCfxBiomesBatchRefusalCase(uint64_t seed, CaseStats *total,
    int *fail)
{
    Generator g;
    setupGenerator(&g, MC_1_21_4, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    // gpu.c's GPU_MAX_POINTS dispatch cap (kept private there).
    enum { CAP_POINTS = 1 << 20 };
    const int bx[2] = { 0, 16 };
    const int bz[2] = { 0, 16 };
    const int w[2] = { CAP_POINTS, 1 };
    const int rows[2] = { 1, 1 };
    const int st[2] = { 4, 4 };
    const long offs[2] = { 0, 1 };
    const int y = 64;

    int failMask = (int) 0x5A5A5A5A;
    int tiny = 0;
    int rc = gpuCfxSurfaceBiomesBatch(&failMask, &tiny, offs, &g.bn, g.sha,
        bx, bz, w, rows, st, &y, 2);
    if (rc == 0)
    {
        printf("  FAIL: over-cap biomes batch was attempted\n");
        *fail = 1;
    }
    else if (failMask != (int) 0x5A5A5A5A)
    {
        printf("  FAIL: over-cap biomes refusal touched failMask (%#x)\n",
            failMask);
        *fail = 1;
    }
    else
    {
        printf("  biomes batch refusal: cap+1 points rejected before any"
            " write\n");
    }
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
        float y[4];
        int ids[4];
        int hi[4], yi[4];
        if (gpuRuntimeInitialized())
        {
            printf("FAIL: runtime initialized before any dispatch in off"
                " mode\n");
            return 1;
        }
        mapOverworldSurfaceBiome(ids, &probe.bn, probe.sha, 4, 0, 0, 1, 1);
        gpuTryHeightHook(y, ids, &probe.bn, 0, 0, 1, 1);
        // the batched strided-rows entries share the off contract
        gpuMapOverworldSurfaceBiomeRows(ids, &probe.bn, probe.sha, 4,
            0, 0, 2, 2, 16);
        gpuMapOverworldTerrainHeightRows(y, ids, &probe.bn, 0, 0, 2, 2, 16);
        // ... and so do the bit-exact cfx product dispatches
        if (gpuCfxOverviewHeights(hi, &probe.bn, 0, 0, 1, 1, 1, 1,
                0, 0, 0) == 0
            || gpuCfxSurfaceBiomes(ids, &probe.bn, probe.sha, 0, 0, 1, 1, 1,
                yi) == 0)
        {
            printf("FAIL: cfx dispatch succeeded in off mode\n");
            return 1;
        }
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

    // z-stride row batching: stride 16 is the mod's overview-row shape,
    // 7 and 33 are off-parity strides that catch z-decode bugs. A positive,
    // a negative, a zero-straddling and two far origins cover the buckets;
    // both biome scales; heights ride on every case; the oldest and newest
    // biome trees get one parity probe each.
    runZStrideCase(mainMc, 0, seeds[0], 100000, 100000, 96, 8, 16, 4,
        &total, &fail);
    runZStrideCase(mainMc, 0, seeds[0], -100000, -100000, 96, 8, 7, 4,
        &total, &fail);
    runZStrideCase(mainMc, 0, seeds[0], -48, -48, 64, 6, 33, 1,
        &total, &fail);
    runZStrideCase(mainMc, 0, seeds[0], 1000000, 1000000, 96, 8, 33, 4,
        &total, &fail);
    runZStrideCase(mainMc, 0, seeds[0], 7500000, 7500000, 64, 6, 16, 4,
        &total, &fail);
    runZStrideCase(MC_1_18, 0, seeds[0], 0, 0, 96, 8, 7, 4, &total, &fail);
    runZStrideCase(MC_26_3, 0, seeds[0], 0, 0, 96, 8, 33, 4, &total, &fail);

    // cfx product parity: the mod's shim hot path re-run bit-exactly on the
    // fp64 kernels. Default: the main version over the first three origins
    // (origin straddle, far field, negative quadrant) with every covered
    // LOD shape; --full extends to all seeds, the world-border origins and
    // the oldest/newest biome-tree versions plus a large-biomes case. Any
    // single mismatch is a porting bug - there are no drift budgets here.
    const int cfxShapeCnt = CFX_LSHAPE_CNT;
    for (int si = 0; si < (full ? seedCnt : 1); si++)
    {
        const int cfxOriginCnt = full
            ? (int)(sizeof(origins) / sizeof(origins[0]))
            : 3;
        for (int oi = 0; oi < cfxOriginCnt; oi++)
            runCfxCase(mainMc, 0, seeds[si], origins[oi][0], origins[oi][1],
                cfxShapeCnt, &total, &fail);
    }
    if (full)
    {
        runCfxCase(MC_1_18, 0, seeds[0], 0, 0, 2, &total, &fail);
        runCfxCase(MC_26_3, 0, seeds[0], 1000000, 1000000, 2, &total, &fail);
        runCfxCase(mainMc, LARGE_BIOMES, seeds[0], 100000, 100000, 1,
            &total, &fail);
    }
    runCfxSlabRotationCase(seeds[0], &total, &fail);
    runCfxHeightsBatchCase(seeds[0], &total, &fail);
    runCfxHeightsBatchRefusalCase(seeds[0], &total, &fail);
    runCfxBiomesBatchCase(seeds[0], &total, &fail);
    runCfxBiomesBatchRefusalCase(seeds[0], &total, &fail);
    runCfxModeCase(seeds[0], &total, &fail);

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
