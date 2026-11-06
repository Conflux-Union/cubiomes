/*
 * CPU-vs-GPU parity harness for the 1.18+ Overworld surface biome GPU
 * engine (gpu.c). Build and run via `make gpu-test`.
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
        mapOverworldTerrainHeight(height, 0, bn, x, z, w, h);
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
    mapOverworldTerrainHeight(height, 0, bn, x4min, z4min, sw, sh);
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
            "(id %lld, np %lld, max np delta %lld)\n",
            total.idMismatch + total.npMismatch, total.pts,
            total.idMismatch, total.npMismatch,
            (long long) total.maxNpDelta);
        return 1;
    }
    printf("PARITY OK: %lld points, 0 mismatches\n", total.pts);
    return 0;
}
