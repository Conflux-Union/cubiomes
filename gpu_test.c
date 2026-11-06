/*
 * CPU-vs-GPU parity harness for the 1.18+ Overworld surface biome and
 * terrain-height GPU engine (gpu.c). Build and run via `make gpu-test`.
 *
 * The CPU side is the reference itself: the real mapOverworldSurfaceBiome()
 * wrapper guards the harness's own replication (which additionally exposes
 * the climate np values), and the GPU dispatch must match both bit-exactly.
 * The engine is a pure executor, so any mismatch is a defect; the harness
 * exits nonzero and prints a mismatch table in that case. Without a usable
 * OpenCL fp64 device it skips cleanly with exit 0.
 *
 * Matrix: the default run covers one seed over all origins and shapes on the
 * main version, a reduced origin set for every supported biome-tree version,
 * and one large-biomes case. `--full` extends the main-version sweep to all
 * seeds (>= 3 seeds x origins {0, 100k, 1M} quart x world-border extreme
 * (quart 7.5M) x scale {1, 4} dense x scale-4 strided rows). Each case prints
 * its header before it runs and its timing after, so a stall names the case
 * and the phase that caused it.
 *
 * Height sections exercise the terrain overview heights the same way: the
 * guarded dispatch (gpuTryHeightHook) against the CPU reference over the
 * consumer query shapes (dense quart maps, the mod's strided overview rows,
 * 4x4 chunk tiles, isolated single columns), the transparent
 * mapOverworldTerrainHeight()/mapApproxHeight() entries in off/auto/force
 * with engagement counting, and a row-vs-isolated shape-invariance case.
 * Heights and the ids at the surface are compared bit-exact.
 *
 * A transparent-path section exercises the public mapOverworldSurfaceBiome()
 * entry (the guarded GPU hook) in off/auto/force modes: every pass must match
 * the CPU reference, auto/force must engage the engine on every public call
 * (dispatch counter, CUBIOMES_ENABLE_GPU builds), and a row-vs-isolated case
 * asserts query-shape invariance through the entry.
 */

#include "gpu.h"

#include "biomenoise.h"
#include "biomes.h"
#include "generator.h"
#include "layers.h"
#include "rng.h"
#include "util.h"

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

typedef struct
{
    long long pts;
    long long idMismatch;
    long long npMismatch;
    long long heightMismatch;
    int64_t maxNpDelta;
    long long refReplicationMismatch;
    double tWrap, tRef, tGpu;
} CaseStats;

static void reportCase(const CaseStats *st)
{
    printf("  -> %s: %lld pts, id mismatch %lld, np mismatch %lld "
        "(max delta %lld) [wrap %.1fs ref %.1fs gpu %.1fs]\n",
        st->idMismatch == 0 && st->npMismatch == 0
        && st->refReplicationMismatch == 0 ? "ok" : "FAIL",
        st->pts, st->idMismatch, st->npMismatch,
        (long long) st->maxNpDelta,
        st->tWrap, st->tRef, st->tGpu);
    fflush(stdout);
}

/*
 * Runs one (version, seed, origin, shape) case:
 * per row (dense = a single row), compare
 *   - harness reference replication vs the real wrapper (self-check)
 *   - GPU dispatch vs the reference (ids + np)
 */
static void runCase(int mc, uint32_t flags, uint64_t seed, int ox, int oz,
    int shapeIdx, CaseStats *st)
{
    const Shape *sh = &SHAPES[shapeIdx];
    printf("case mc=%d seed=%llu origin=(%d,%d) %s\n",
        mc, (unsigned long long) seed, ox, oz, shapeNameByIdx(shapeIdx));
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
        // CPU golden: the real wrapper over the dense row
        mapOverworldSurfaceBiome(wrapIds, &g.bn, g.sha, sh->scale,
            ox, rowZ, rawW, 1);
        double t1 = nowSeconds();
        // harness reference (ids must match the wrapper bit-exactly)
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
                st->refReplicationMismatch++;
                if (st->refReplicationMismatch <= 3)
                    printf("  ref-vs-wrapper @(%d,%d): ref=%d wrapper=%d\n",
                        ox + i * stride, rowZ, refIds[idx], wrapIds[idx]);
            }
            if (gpuIds[idx] != refIds[idx])
            {
                st->idMismatch++;
                if (st->idMismatch <= 3)
                    printf("  id @(%d,%d): gpu=%d cpu=%d\n",
                        ox + i * stride, rowZ, gpuIds[idx], refIds[idx]);
            }
            const int64_t *cnp = refNp + (size_t)idx * NP_CH;
            const int64_t *gnp = gpuNp + (size_t)idx * NP_CH;
            for (int k = 0; k < NP_CH; k++)
            {
                if (gnp[k] != cnp[k])
                {
                    st->npMismatch++;
                    int64_t d = npDelta(gnp[k], cnp[k]);
                    if (d > st->maxNpDelta)
                        st->maxNpDelta = d;
                    if (st->npMismatch <= 3)
                        printf("  np[%d] @(%d,%d): gpu=%lld cpu=%lld\n",
                            k, ox + i * stride, rowZ,
                            (long long) gnp[k], (long long) cnp[k]);
                }
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
    runCase(mc, flags, seed, ox, oz, shapeIdx, &st);
    reportCase(&st);
    if (st.idMismatch || st.npMismatch || st.refReplicationMismatch)
        *fail = 1;
    total->pts += st.pts;
    total->idMismatch += st.idMismatch;
    total->npMismatch += st.npMismatch;
    total->refReplicationMismatch += st.refReplicationMismatch;
    if (st.maxNpDelta > total->maxNpDelta)
        total->maxNpDelta = st.maxNpDelta;
}

//----------------------------------------------------------------------
// Transparent dispatch path: the public mapOverworldSurfaceBiome() entry,
// which tries the guarded GPU hook before its CPU body when compiled with
// CUBIOMES_ENABLE_GPU. Every pass must reproduce the CPU reference
// bit-exactly; auto/force must engage the engine on every public call
// (dispatch counter), off must not touch it. Points are counted once per
// case even though all three passes compare them.

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

static void reportTransparent(const char *mode, long long idMismatch,
    uint64_t engaged, uint64_t expected)
{
    printf("  -> %s %s: id mismatch %lld [gpu %llu/%llu]\n",
        mode,
        idMismatch == 0 && engaged == expected ? "ok" : "FAIL",
        idMismatch, (unsigned long long) engaged,
        (unsigned long long) expected);
    fflush(stdout);
}

static void runTransparentCase(int mc, uint32_t flags, uint64_t seed, int ox,
    int oz, int shapeIdx, CaseStats *total, int *fail)
{
    const Shape *sh = &TSHAPES[shapeIdx];
    printf("transparent case mc=%d seed=%llu origin=(%d,%d) %s\n",
        mc, (unsigned long long) seed, ox, oz, tshapeNameByIdx(shapeIdx));
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
    long long idMismatch = 0;
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
                    idMismatch++;
                    if (idMismatch <= 3)
                        printf("  %s id @(%d,%d): pub=%d ref=%d\n",
                            modeNames[p], ox + i, rowZ, pubIds[i], refIds[i]);
                }
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
        reportTransparent(modeNames[p], idMismatch, engaged, expected);
    }
    gpuSetMode(modeBefore);

    free(refIds);
    free(pubIds);
    free(refNp);
    total->pts += (long long) sh->w * sh->h;
    total->idMismatch += idMismatch;
    if (idMismatch || shapeFail)
        *fail = 1;
}

/*
 * Query-shape invariance through the public entry: one strided/dense row
 * computed by a single GPU dispatch must equal isolated single-point public
 * queries (each its own dispatch) and the CPU reference row.
 */
static void runTransparentShapeCase(int mc, uint32_t flags, uint64_t seed,
    int ox, int oz, CaseStats *total, int *fail)
{
    printf("transparent shape case mc=%d seed=%llu origin=(%d,%d) "
        "row-vs-isolated\n", mc, (unsigned long long) seed, ox, oz);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);

    const int modeBefore = gpuGetMode();
    gpuSetMode(GPU_MODE_AUTO);
    const uint64_t countBefore = gpuDispatchCount();
    uint64_t calls = 0;
    long long idMismatch = 0;
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
        long long scaleMismatch = 0;
        for (int i = 0; i < w; i += step)
        {
            int one;
            mapOverworldSurfaceBiome(&one, &g.bn, g.sha, scale,
                ox + i, oz, 1, 1);
            calls++;
            if (one != row[i])
            {
                scaleMismatch++;
                if (scaleMismatch <= 3)
                    printf("  scale%d isolated @(%d,%d): one=%d row=%d\n",
                        scale, ox + i, oz, one, row[i]);
            }
            if (row[i] != refIds[i])
            {
                scaleMismatch++;
                if (scaleMismatch <= 3)
                    printf("  scale%d ref @(%d,%d): row=%d ref=%d\n",
                        scale, ox + i, oz, row[i], refIds[i]);
            }
            pts++;
        }
        printf("  -> scale%d: %lld sampled pts, id mismatch %lld\n",
            scale, pts, scaleMismatch);
        fflush(stdout);
        idMismatch += scaleMismatch;
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
    total->idMismatch += idMismatch;
    if (idMismatch || shapeFail)
        *fail = 1;
}

/*
 * Reseed regression for the table cache: the engine keys its packed tables
 * on a fingerprint of the seed-derived state, because setBiomeSeed()
 * re-derives the tables in place and pointer identity cannot see a reseed.
 * Public-entry calls on the SAME BiomeNoise across a reseed and across a
 * large-biomes flip (same seed, same sha) must each match the CPU reference;
 * a stale cache would repeat the first configuration's tables.
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
    long long idMismatch = 0;
    for (int p = 0; p < 3; p++)
    {
        setupGenerator(&g, mc, passes[p].flags);
        applySeed(&g, DIM_OVERWORLD, passes[p].seed);
        cpuSurfaceBiomeRef(refIds, refNp, &g.bn, g.sha, 4, 0, 0, w, 1);
        mapOverworldSurfaceBiome(pubIds, &g.bn, g.sha, 4, 0, 0, w, 1);
        for (int i = 0; i < w; i++)
        {
            if (pubIds[i] != refIds[i])
            {
                idMismatch++;
                if (idMismatch <= 3)
                    printf("  pass %d id @(%d,0): pub=%d ref=%d\n",
                        p, i, pubIds[i], refIds[i]);
            }
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

    printf("  -> %s: id mismatch %lld\n",
        idMismatch == 0 && !shapeFail ? "ok" : "FAIL", idMismatch);
    fflush(stdout);
    free(pubIds);
    free(refIds);
    free(refNp);
    total->pts += 3 * w;
    total->idMismatch += idMismatch;
    if (idMismatch || shapeFail)
        *fail = 1;
}

//----------------------------------------------------------------------
// Terrain overview heights: the guarded dispatch (gpuTryHeightHook) and the
// transparent mapOverworldTerrainHeight()/mapApproxHeight() entries, against
// the pure-CPU reference (cpuHeightRef pins the engine off), bit-exact on
// the heights and on the ids at the clamped surface.

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

static void reportHeightCase(const CaseStats *st)
{
    printf("  -> %s: %lld pts, height mismatch %lld, id mismatch %lld "
        "[cpu %.1fs gpu %.1fs]\n",
        st->heightMismatch == 0 && st->idMismatch == 0 ? "ok" : "FAIL",
        st->pts, st->heightMismatch, st->idMismatch, st->tRef, st->tGpu);
    fflush(stdout);
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
    printf("height case mc=%d seed=%llu origin=(%d,%d) %s\n",
        mc, (unsigned long long) seed, ox, oz, hshapeNameByIdx(shapeIdx));
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

    CaseStats st = {0};
    const int calls = stride == 1 ? 1 : sh->h;
    for (int c = 0; c < calls; c++)
    {
        const int rowZ = oz + c * stride;
        const int rw = stride == 1 ? sh->w : rawW;
        const int rh = stride == 1 ? sh->h : 1;
        st.tRef += cpuHeightRef(refY, refIds, &g.bn, ox, rowZ, rw, rh);
        double t0 = nowSeconds();
        if (gpuTryHeightHook(gpuY, gpuIds, &g.bn, ox, rowZ, rw, rh))
        {
            fprintf(stderr, "gpu-test: GPU height dispatch failed at mc=%d "
                "seed=%llu (%d,%d) - engine reported unavailable\n",
                mc, (unsigned long long) seed, ox, rowZ);
            exit(2);
        }
        double t1 = nowSeconds();
        st.tGpu += t1 - t0;

        for (int j = 0; j < rh; j++)
        {
            for (int i = 0; i < rw; i += stride)
            {
                const int idx = j * rw + i;
                st.pts++;
                if (gpuY[idx] != refY[idx])
                {
                    st.heightMismatch++;
                    if (st.heightMismatch <= 3)
                        printf("  height @(%d,%d): gpu=%f cpu=%f\n",
                            ox + i, rowZ + j, (double) gpuY[idx],
                            (double) refY[idx]);
                }
                if (sh->ids && gpuIds[idx] != refIds[idx])
                {
                    st.idMismatch++;
                    if (st.idMismatch <= 3)
                        printf("  id @(%d,%d): gpu=%d cpu=%d\n",
                            ox + i, rowZ + j, gpuIds[idx], refIds[idx]);
                }
            }
        }
    }

    reportHeightCase(&st);
    free(refY);
    free(gpuY);
    free(refIds);
    free(gpuIds);
    total->pts += st.pts;
    total->heightMismatch += st.heightMismatch;
    total->idMismatch += st.idMismatch;
    if (st.heightMismatch || st.idMismatch)
        *fail = 1;
}

/*
 * Transparent height dispatch through the public mapOverworldTerrainHeight()
 * entry in off/auto/force: every pass must match the pure-CPU reference,
 * auto/force must engage the engine on every public call
 * (gpuHeightDispatchCount, CUBIOMES_ENABLE_GPU builds), off must not touch
 * it. The auto pass also re-queries sampled points as isolated single
 * columns: one strided/dense row dispatch must equal the isolated calls.
 */
static void runTransparentHeightCase(int mc, uint32_t flags, uint64_t seed,
    int ox, int oz, CaseStats *total, int *fail)
{
    const int w = 64;
    const int rows = 2;
    printf("transparent height case mc=%d seed=%llu origin=(%d,%d) "
        "w=%d rows=%d + isolated\n",
        mc, (unsigned long long) seed, ox, oz, w, rows);
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
    long long heightMismatch = 0, idMismatch = 0;
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
                if (pubY[idx] != refY[idx])
                {
                    heightMismatch++;
                    if (heightMismatch <= 3)
                        printf("  %s height @(%d,%d): pub=%f ref=%f\n",
                            modeNames[p], ox + i, oz + j,
                            (double) pubY[idx], (double) refY[idx]);
                }
                if (pubIds[idx] != refIds[idx])
                {
                    idMismatch++;
                    if (idMismatch <= 3)
                        printf("  %s id @(%d,%d): pub=%d ref=%d\n",
                            modeNames[p], ox + i, oz + j,
                            pubIds[idx], refIds[idx]);
                }
            }
        }
        printf("  -> %s: height mismatch %lld, id mismatch %lld "
            "[gpu %llu/%llu]\n", modeNames[p], heightMismatch, idMismatch,
            (unsigned long long) engaged, (unsigned long long) expected);
        fflush(stdout);
    }

    // Shape invariance: isolated single-column queries through the same
    // public entry must reproduce the row dispatch's heights and ids.
    gpuSetMode(GPU_MODE_AUTO);
    for (int i = 0; i < w; i += 7)
    {
        float oneY;
        int oneId;
        mapOverworldTerrainHeight(&oneY, &oneId, &g.bn, ox + i, oz, 1, 1);
        if (oneY != pubY[i] || oneId != pubIds[i])
        {
            heightMismatch++;
            if (heightMismatch <= 3)
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
    total->heightMismatch += heightMismatch;
    total->idMismatch += idMismatch;
    if (heightMismatch || idMismatch || shapeFail)
        *fail = 1;
}

/*
 * The Generator-facing consumer entry mapApproxHeight(): on 1.18+ Overworld
 * it gates nptype and delegates to mapOverworldTerrainHeight(). The tile and
 * single-column shapes real consumers use, in auto mode, against the
 * mode-off CPU reference.
 */
static void runApproxHeightCase(int mc, uint32_t flags, uint64_t seed,
    int ox, int oz, CaseStats *total, int *fail)
{
    printf("transparent mapApproxHeight case mc=%d seed=%llu origin=(%d,%d) "
        "tile4x4 + single\n", mc, (unsigned long long) seed, ox, oz);
    fflush(stdout);

    Generator g;
    setupGenerator(&g, mc, flags);
    applySeed(&g, DIM_OVERWORLD, seed);
    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_OVERWORLD, g.seed);

    const int w = 4, h = 4;
    float refY[16], pubY[16];
    int refIds[16], pubIds[16];
    long long heightMismatch = 0, idMismatch = 0;

    cpuHeightRef(refY, refIds, &g.bn, ox, oz, w, h);
    mapApproxHeight(pubY, pubIds, &g, &sn, ox, oz, w, h);
    for (int i = 0; i < w * h; i++)
    {
        if (pubY[i] != refY[i])
        {
            heightMismatch++;
            if (heightMismatch <= 3)
                printf("  tile height @(%d,%d): pub=%f ref=%f\n",
                    ox + i % w, oz + i / w, (double) pubY[i],
                    (double) refY[i]);
        }
        if (pubIds[i] != refIds[i])
        {
            idMismatch++;
            if (idMismatch <= 3)
                printf("  tile id @(%d,%d): pub=%d ref=%d\n",
                    ox + i % w, oz + i / w, pubIds[i], refIds[i]);
        }
    }

    float oneRefY, onePubY;
    int oneRefId, onePubId;
    cpuHeightRef(&oneRefY, &oneRefId, &g.bn, ox + 2, oz + 2, 1, 1);
    mapApproxHeight(&onePubY, &onePubId, &g, &sn, ox + 2, oz + 2, 1, 1);
    if (onePubY != oneRefY || onePubId != oneRefId)
    {
        heightMismatch++;
        printf("  single @(%d,%d): pub=(%f,%d) ref=(%f,%d)\n",
            ox + 2, oz + 2, (double) onePubY, onePubId,
            (double) oneRefY, oneRefId);
    }

    printf("  -> %s: height mismatch %lld, id mismatch %lld\n",
        heightMismatch == 0 && idMismatch == 0 ? "ok" : "FAIL",
        heightMismatch, idMismatch);
    fflush(stdout);
    total->pts += (long long) w * h + 1;
    total->heightMismatch += heightMismatch;
    total->idMismatch += idMismatch;
    if (heightMismatch || idMismatch)
        *fail = 1;
}

int main(int argc, char **argv)
{
    int full = 0;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--full"))
            full = 1;
        else
        {
            fprintf(stderr, "usage: gpu_test [--full]\n");
            return 2;
        }
    }

    Generator probe;
    setupGenerator(&probe, MC_1_21_4, 0);
    applySeed(&probe, DIM_OVERWORLD, 1ULL);
    if (gpuGetMode() == GPU_MODE_OFF)
    {
        printf("skip: CUBIOMES_GPU=off\n");
        return 0;
    }
    if (!gpuAvailable(&probe))
    {
        printf("skip: no usable OpenCL fp64 device (CUBIOMES_GPU=%s)\n",
            gpuGetMode() == GPU_MODE_FORCE ? "force" : "auto");
        return 0;
    }
    printf("OpenCL device: %s\n", gpuDeviceName());
    printf("parity matrix: %s (deterministic flags CPU reference vs GPU, "
        "bit-exact required)\n\n", full ? "FULL" : "default (--full extends)");
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
            if (origins[oi][0] != 0 && !(full && origins[oi][0] == 7500000))
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
            if (origins[oi][0] != 0 && !(full && origins[oi][0] == 7500000))
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
    if (total.refReplicationMismatch)
    {
        printf("HARNESS BROKEN: reference replication mismatches the "
            "wrapper: %lld\n", total.refReplicationMismatch);
        return 2;
    }
    if (fail)
    {
        printf("PARITY FAIL: %lld / %lld points mismatched "
            "(id %lld, np %lld, height %lld, max np delta %lld)\n",
            total.idMismatch + total.npMismatch + total.heightMismatch,
            total.pts, total.idMismatch, total.npMismatch,
            total.heightMismatch, (long long) total.maxNpDelta);
        return 1;
    }
    printf("PARITY OK: %lld points, 0 mismatches\n", total.pts);
    return 0;
}
