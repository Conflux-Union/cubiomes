#ifndef GPU_H_
#define GPU_H_

/*
 * Optional OpenCL GPU execution for the 1.18+ Overworld climate biome
 * sampling and the terrain overview heights. The GPU engine mirrors
 * mapOverworldSurfaceBiome() + depth-aware sampleBiomeNoise() and
 * mapOverworldTerrainHeight() semantics, evaluated in single precision: the
 * CPU reference computes its noise chains in double, so GPU outputs carry a
 * coordinate-dependent drift against it (negligible near the origin, growing
 * toward the world border; see the GPU section of README.md). A drift-budget
 * harness (gpu_test.c, `make gpu-test`) gates every change: biome-id flip
 * rates and height/climate delta budgets per distance bucket, plus exact
 * checks for everything that must stay deterministic (query-shape
 * invariance, mode handling, CPU self-consistency).
 *
 * Tables are packed from the live Generator/BiomeNoise structs (perm tables,
 * per-octave noise parameters, spline trees, biome tree), so all 1.18+
 * versions come free without per-version hardcoded tables in this layer. One
 * packed table set serves both the biome and the height kernel, keyed on one
 * generator state at a time: dispatches that alternate between generators
 * (e.g. two seeds) re-pack the tables on every call - correct, but slow, so
 * batch per generator before switching.
 *
 * Thread model: one process-global engine with ONE command queue. A mutex
 * serializes submission (table readiness, buffer growth, kernel and
 * readback enqueues); the readback wait happens outside the lock against
 * one of two rotating output slabs, so the next dispatch enqueues into the
 * other slab while the previous one drains - concurrent callers overlap
 * instead of queueing up, but never race. A wedged GPU hangs the calling
 * thread (the promise is that GPU calls never fail or change results,
 * never that they cannot hang).
 * Consumers that link this module need the dynamic loader
 * (-ldl on older glibc) and threads (-lpthread on older glibc).
 *
 * Mode handling: the environment variable CUBIOMES_GPU (off|auto|force) is
 * parsed at first use; gpuSetMode() overrides it. "auto" (default) uses the
 * GPU when available and silently falls back to the CPU otherwise; "off"
 * never touches the GPU - it does not even load the OpenCL runtime; "force"
 * is a debug aid that reports loudly when the GPU is unavailable. Read-only
 * queries (gpuGetMode/gpuAvailable) never fail because of the GPU; the
 * dispatches below return nonzero on a GPU error and latch the engine dead
 * for the process so later calls fall back too.
 *
 * Transparent dispatch: when a consumer builds this module together with
 * CUBIOMES_ENABLE_GPU, mapOverworldSurfaceBiome() (biomenoise.c) tries the
 * GPU hook below before its CPU body, and so does mapOverworldTerrainHeight()
 * for the terrain overview heights. The generic flat-Y queries
 * (genBiomes()/getBiomeAt()) deliberately stay on the CPU: the kernel derives
 * the terrain surface per column, which is a different contract than the
 * caller-chosen Y of those queries.
 */

#include "generator.h"

enum
{
    GPU_MODE_OFF = 0,
    GPU_MODE_AUTO = 1,
    GPU_MODE_FORCE = 2,
};

/* Override the GPU execution mode (default: CUBIOMES_GPU env or auto).
 * Invalid values are ignored. */
void gpuSetMode(int mode);

/* Returns the effective mode (GPU_MODE_OFF, GPU_MODE_AUTO or GPU_MODE_FORCE). */
int gpuGetMode(void);

/*
 * Cheap health check for GPU dispatch of the given generator: an OpenCL
 * GPU device is present and healthy, the generator is an unseeded-special
 * 1.18+ Overworld biome-noise generator (the supported version/dimension
 * combination). Scale coverage is fixed: the dispatch supports the surface
 * scales 1 (block + voronoi) and 4 (quart). Does not validate the requested
 * area; the dispatch re-checks its own arguments.
 */
int gpuAvailable(const Generator *g);

/* Name of the probed OpenCL device, or an empty string before first use /
 * when unavailable. Diagnostic aid, not part of the dispatch contract. */
const char *gpuDeviceName(void);

/* Returns nonzero once the engine has initialized the OpenCL runtime
 * (loader load + device probe + program build). Diagnostic aid for tests:
 * OFF mode must leave this at zero for the whole process lifetime. */
int gpuRuntimeInitialized(void);

/* Number of surface-biome dispatches the engine has completed successfully
 * (kernel run + results read back), process-global. Diagnostic aid for
 * verifying that a code path actually engaged the GPU. Height dispatches are
 * counted separately by gpuHeightDispatchCount(). */
uint64_t gpuDispatchCount(void);

/* Number of terrain-height dispatches the engine has completed
 * successfully, process-global. Diagnostic aid, same contract as
 * gpuDispatchCount(). */
uint64_t gpuHeightDispatchCount(void);

/*
 * The guarded GPU dispatch behind mapOverworldSurfaceBiome() when compiled
 * with CUBIOMES_ENABLE_GPU. This is the single place that owns the dispatch
 * guard; it returns 0 only when the GPU served the whole area (the
 * float-precision approximation of the wrapper's CPU result) and nonzero
 * whenever the caller must run the CPU
 * path - unsupported arguments, disabled/dead engine, or any GPU error.
 * gpuMapOverworldSurfaceBiome() is the Generator-facing form of the same
 * dispatch and shares this guard.
 */
int gpuTrySurfaceBiomeHook(int *out, const BiomeNoise *bn, uint64_t sha,
    int scale, int x, int z, int w, int h);

/*
 * GPU approximation of mapOverworldSurfaceBiome(out, &g->bn, sha, scale, x,
 * z, w, h) in float precision (see the header comment for the drift
 * contract). Returns 0 on success; nonzero when the GPU engine is disabled,
 * dead, or the arguments are unsupported (mirrors the wrapper's error
 * convention so callers fall back to the CPU path). When `np` is non-NULL it
 * receives NP_MAX climate noise parameters per point (int64_t[np? w*h :
 * ...][6]), which the CPU wrapper does not expose; values are the GPU's own
 * quantized parameters for the same point, close to but not identical to
 * sampleBiomeNoise()'s `np` output.
 */
int gpuMapOverworldSurfaceBiome(int *out, const Generator *g, uint64_t sha,
    int scale, int x, int z, int w, int h, int64_t *np);

/*
 * Batched strided-rows form of the surface-biome dispatch: computes
 * `rows` separate w-wide biome rows in ONE GPU dispatch, row j at
 * z = z + j*zStride (so the contiguous h-row block of
 * gpuMapOverworldSurfaceBiome() is the zStride == 1 case), each written to
 * out[j*w .. j*w+w-1] row-major. Consumers that stream strided overview
 * rows row by row pay one dispatch per row, which collapses occupancy when
 * a row alone cannot fill the device; this entry folds the whole row set
 * into a single dispatch with identical per-point results. Returns 0 when
 * the GPU served all rows*W points (float-precision approximation, same
 * drift contract as above) and nonzero when the caller must run the CPU
 * path per row - unsupported arguments (non-positive w/rows/zStride,
 * scale other than 1 or 4, rows*w over the ~1M-point dispatch cap,
 * pre-1.18 versions), disabled/dead engine, or any GPU error. Climate np
 * values are not exposed here; use gpuMapOverworldSurfaceBiome() if the
 * caller needs them.
 */
int gpuMapOverworldSurfaceBiomeRows(int *out, const BiomeNoise *bn,
    uint64_t sha, int scale, int x, int z, int w, int rows, int zStride);

/*
 * The guarded GPU dispatch behind mapOverworldTerrainHeight() when compiled
 * with CUBIOMES_ENABLE_GPU, the height-path twin of
 * gpuTrySurfaceBiomeHook(). Returns 0 only when the GPU served the whole
 * area (float-precision approximation of the CPU heights): y[] receives the
 * quart-scale surface heights and, when ids is non-NULL, ids[] the
 * surface-biome ids at the clamped surface. Returns nonzero whenever the
 * caller must run the CPU path - unsupported arguments (NULL y, non-positive
 * or oversized area, a BiomeNoise not fully initialized by setBiomeSeed(),
 * pre-1.18 versions), disabled/dead engine, or any GPU error. Block-scale
 * strided height queries (mapOverworldTerrainHeightBlocks and friends) are
 * not covered and stay on the CPU.
 */
int gpuTryHeightHook(float *y, int *ids, const BiomeNoise *bn,
    int x, int z, int w, int h);

/*
 * Batched strided-rows form of the terrain-height dispatch, the height twin
 * of gpuMapOverworldSurfaceBiomeRows(): `rows` w-wide height rows, row j at
 * z = z + j*zStride, one GPU dispatch for the whole set, each row written to
 * y[j*w .. j*w+w-1] (and ids[j*w ..] when ids is non-NULL) row-major.
 * Returns 0 when the GPU served all rows*w points (float-precision
 * approximation, same drift contract as the guarded hook) and nonzero when
 * the caller must run the CPU path per row - unsupported arguments
 * (NULL y, non-positive w/rows/zStride, rows*w over the ~1M-point dispatch
 * cap, a BiomeNoise not fully initialized by setBiomeSeed(), pre-1.18
 * versions), disabled/dead engine, or any GPU error.
 */
int gpuMapOverworldTerrainHeightRows(float *y, int *ids, const BiomeNoise *bn,
    int x, int z, int w, int rows, int zStride);

/*
 * Bit-exact dispatches for the mod's product hot path (confluxnative.c).
 * Unlike the float approximations above, these re-run the shim's own CPU
 * chains in double precision and must stay bit-identical to them: the shim
 * compiles with -ffp-contract=off, the kernels run with FP_CONTRACT OFF,
 * and the packed tables carry the seed-derived noise parameters at full
 * double fidelity. They need an fp64 OpenCL device (cl_khr_fp64); without
 * one they return nonzero on every call and the caller stays on the CPU.
 * Mode handling, engine latching and table caching are shared with the
 * float dispatches; the counters count by family (height vs biome), not
 * per kernel flavor.
 */

/*
 * The shim's cfxModernOverviewHeight() over a rectangular quart lattice:
 * heights[row*gridW + col] receives the solver result for quart cell
 * (x4 + col*stepX4, z4 + row*stepZ4). The caller owns the lattice shape
 * (macro grid with step 1 for the shim's stride<4 path, stride/4 steps for
 * its direct path) and the cheap integer interpolation between lattice
 * points. mainFirst/minMaxFirst/truncated are the shim's
 * cfxNoiseDetailForStride() result for the sampling stride. Returns 0 when
 * the GPU filled all gridW*gridH heights bit-exactly; nonzero means run
 * the CPU path (unsupported arguments, disabled/dead engine, no fp64
 * device, or any GPU error).
 */
int gpuCfxOverviewHeights(int *heights, const BiomeNoise *bn,
    int x4, int z4, int gridW, int gridH, int stepX4, int stepZ4,
    int mainFirst, int minMaxFirst, int truncated);

/*
 * Batch variant of gpuCfxOverviewHeights(): `n` independent lattices in ONE
 * submission, so a wave of tiles pays one lock acquisition and one readback
 * wait instead of n. Block i solves the lattice gpuCfxOverviewHeights()
 * would for (x4[i], z4[i], gridW[i] x gridH[i], stepX4[i], stepZ4[i]) and
 * writes gridW[i]*gridH[i] ints at heights[flatOffsets[i]]: heights is
 * caller-provided flat storage, flatOffsets[i] is its int index for block i
 * (non-overlapping; the caller owns the layout). The detail scalars are one
 * set shared by every block. n must not exceed 32 (*failMask is an int
 * bitmask) and the sum of gridW[i]*gridH[i] is capped like the single
 * entry. Returns 0 when the batch was attempted: blocks whose enqueue
 * failed host-side carry bit i in *failMask, their output range is left
 * untouched and the rest is filled bit-exactly, so the caller CPU-repairs
 * exactly those blocks. Nonzero means the whole batch was refused before
 * any GPU work or output write (bad arguments incl. the caps, disabled/
 * dead engine, no fp64 device), *failMask is untouched, and the caller
 * falls back per block. Bit-exact against dispatching the same blocks
 * one by one.
 */
int gpuCfxOverviewHeightsBatch(int *failMask, int *heights,
    const long *flatOffsets, const BiomeNoise *bn, const int *x4,
    const int *z4, const int *gridW, const int *gridH, const int *stepX4,
    const int *stepZ4, int n, int mainFirst, int minMaxFirst,
    int truncated);

/*
 * The shim's cfxSurfaceBiomes() inner loop: for each point (blockX +
 * col*stride, blockZ + row*stride), terrainY[row*w + col] (clamped at 62,
 * exactly like the shim) picks the voronoi nucleus and the double climate
 * stack resolves the surface biome id into ids[row*w + col]. terrainY is
 * row-major w*rows and uploads per call. The CPU nucleus memo this may
 * replace is itself bit-identical to the shim's plain loop, so the kernel
 * inherits bit-exactness against both. Batch-size policy (skip tiny refine
 * cells) belongs to the caller; this guard rejects only unsupported shapes
 * (pre-1.18, over-cap batches, engine problems, no fp64 device).
 */
int gpuCfxSurfaceBiomes(int *ids, const BiomeNoise *bn, uint64_t sha,
    int blockX, int blockZ, int w, int rows, int stride,
    const int *terrainY);

#endif
