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
 * Dispatches for the mod's product hot path (confluxnative.c). Like the
 * float approximations above they re-run the shim's own CPU chains, but
 * in single precision: outputs approximate the double CPU chains
 * coordinate-dependently and are gated by the gpu_test drift harness,
 * not by bit identity. The shim compiles with -ffp-contract=off, the
 * kernels run with FP_CONTRACT OFF, and the same packed tables serve
 * both kernel families. They run on any OpenCL device - no extension
 * requirement. Mode handling, engine latching and table caching are
 * shared with the float dispatches; the counters count by family
 * (height vs biome), not per kernel flavor.
 */

/*
 * The shim's cfxModernOverviewHeight() over a rectangular quart lattice:
 * heights[row*gridW + col] receives the solver result for quart cell
 * (x4 + col*stepX4, z4 + row*stepZ4). The caller owns the lattice shape
 * (macro grid with step 1 for the shim's stride<4 path, stride/4 steps for
 * its direct path) and the cheap integer interpolation between lattice
 * points. mainFirst/minMaxFirst/truncated are the shim's
 * cfxNoiseDetailForStride() result for the sampling stride. Returns 0 when
 * the GPU filled all gridW*gridH heights; nonzero means run the CPU path
 * (unsupported arguments, disabled/dead engine, or any GPU error).
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
 * untouched and the rest is filled by the kernel, so the caller
 * CPU-repairs exactly those blocks. Nonzero means the whole batch was
 * refused before any GPU work or output write (bad arguments incl. the
 * caps, disabled/dead engine), *failMask is untouched, and the caller
 * falls back per block. Bit-exact against dispatching the same blocks
 * one by one (same program, no per-point state).
 */
int gpuCfxOverviewHeightsBatch(int *failMask, int *heights,
    const long *flatOffsets, const BiomeNoise *bn, const int *x4,
    const int *z4, const int *gridW, const int *gridH, const int *stepX4,
    const int *stepZ4, int n, int mainFirst, int minMaxFirst,
    int truncated);

/*
 * The shim's cfxSurfaceBiomes() inner loop: for each point (blockX +
 * col*stride, blockZ + row*stride), terrainY[row*w + col] (clamped at 62,
 * exactly like the shim) picks the voronoi nucleus and the single-precision
 * climate stack resolves the surface biome id into ids[row*w + col].
 * terrainY is row-major w*rows and uploads per call. Batch-size policy
 * (skip tiny refine cells) belongs to the caller; this guard rejects only
 * unsupported shapes (pre-1.18, over-cap batches, engine problems).
 */
int gpuCfxSurfaceBiomes(int *ids, const BiomeNoise *bn, uint64_t sha,
    int blockX, int blockZ, int w, int rows, int stride,
    const int *terrainY);

/*
 * Batch variant of gpuCfxSurfaceBiomes(): `n` independent caller-Y blocks
 * in ONE submission, so a wave of tiles pays one lock acquisition, one
 * table pack, one terrainY upload and one readback wait instead of n.
 * Block i resolves the ids gpuCfxSurfaceBiomes() would for (blockX[i],
 * blockZ[i], w[i] x rows[i], stride[i]) and writes w[i]*rows[i] ints at
 * ids[flatOffsets[i]]: ids is caller-provided flat storage, flatOffsets[i]
 * is its int index for block i (non-overlapping; the caller owns the
 * layout). terrainY is the matching flat input: block i's row-major plane
 * of w[i]*rows[i] ints at flatOffsets[i], uploaded as ONE write into the
 * batch slab's input buffer before the first kernel enqueues. sha and the
 * packed tables are shared by every block. n must not exceed 32 (*failMask
 * is an int bitmask) and the sum of w[i]*rows[i] is capped like the single
 * entry. Returns 0 when the batch was attempted: blocks whose enqueue
 * failed host-side carry bit i in *failMask, their output range is left
 * untouched and the rest is filled by the kernel, so the caller
 * CPU-repairs exactly those blocks. Nonzero means the whole batch was
 * refused before any GPU work or output write (bad arguments incl. the n
 * cap, a negative flatOffsets entry or the points cap; mode off;
 * disabled/dead engine; table-pack failure; slab-growth/allocation
 * failure; the terrainY upload failing to enqueue), *failMask is
 * untouched, and the caller falls back per block. Bit-exact against
 * dispatching the same blocks one by one (same program, no per-point
 * state).
 */
int gpuCfxSurfaceBiomesBatch(int *failMask, int *ids, const long *flatOffsets,
    const BiomeNoise *bn, uint64_t sha,
    const int *blockX, const int *blockZ, const int *w, const int *rows,
    const int *stride, const int *terrainY, int n);

/*
 * Sparse-point form of gpuCfxOverviewHeights(): `n` arbitrary quart points
 * (x4[i], z4[i]) - not a rectangular lattice - solved in ONE kernel launch
 * and read back in ONE read, outY[i] receiving the point's height. The
 * detail scalars are one set shared by every point of the chunk. There is
 * no failMask: a sparse chunk is rejected whole-or-nothing, and rejection
 * fires before ANY GPU work or caller-output write. Returns 0 when the GPU
 * filled all n heights; nonzero means the whole chunk was
 * refused and the caller runs the CPU path per point (NULL pointers, n
 * outside [1, ~1M], detail scalars out of range - mainFirst 0..7,
 * minMaxFirst 0..15, truncated 0/1 - a BiomeNoise not fully initialized by
 * setBiomeSeed() or pre-1.18, mode off, disabled/dead engine,
 * table-pack failure, slab-growth/allocation failure, or an
 * enqueue failure). Bit-exact against the rect entry over the same points.
 */
int gpuCfxOverviewHeightsSparse(int *outY, const BiomeNoise *bn,
    const int *x4, const int *z4, int n,
    int mainFirst, int minMaxFirst, int truncated);

/*
 * Sparse-point form of gpuCfxSurfaceBiomes(): nCells cells, cell j covering
 * edge*edge points spaced `stride` apart starting at (cellX[j], cellZ[j]),
 * resolved in ONE kernel launch and read back in ONE read. The caller's
 * output is one contiguous chunk-local span [0..nCells*edge*edge):
 * outIds[cell*(edge*edge) + row*edge + col] resolves the point
 * (cellX[cell] + col*stride, cellZ[cell] + row*stride), and terrainY packs
 * the caller heights at the same flat index (clamped at 62 by the kernel,
 * exactly like the shim). sha is shared by every point. Like the heights
 * entry there is no failMask: rejection is whole-or-nothing and fires
 * before ANY GPU work or output write. Returns 0 when the GPU filled all
 * nCells*edge*edge ids; nonzero means the whole chunk was
 * refused and the caller runs the CPU path per point (NULL pointers,
 * edge/stride/nCells below 1, the points total over ~1M, a BiomeNoise not
 * fully initialized by setBiomeSeed() or pre-1.18, mode off, disabled/
 * dead engine, table-pack failure, slab-growth/allocation
 * failure, or an enqueue failure). Bit-exact against the rect entry over
 * the same points.
 */
int gpuCfxSurfaceBiomesSparse(int *outIds, const BiomeNoise *bn, uint64_t sha,
    const int *cellX, const int *cellZ,
    const int *terrainY, int edge, int stride, int nCells);

#if 0 /* parked: surface-columns GPU path - net-wall negative in the product seam per the seeded e2e A/B (opening wall +6.2%, first tile +76%); restore = flip the parked guards in both repos and rebuild */
/*
 * The mod's cfxSurfaceColumns() point seam (the exact-columns stage): n
 * anchors, anchor i at block coords (blockX[i], blockZ[i]), all solved in
 * ONE kernel launch and read back in ONE read. The four output planes
 * mirror what mapOverworldSurfaceColumns() fills for the same anchor on
 * MC 1.18+: outSolidY[i] the first solid block of the column's
 * banded surface search, outFluidY[i] the aquifer fluid surface (the
 * INT32_MIN sentinel when no fluid stands above it), outSurfaceY[i] the
 * fluid surface when it stands above the solid one and the solid surface
 * otherwise, and outFlags[i] SURFACE_COLUMN_FLUID iff the fluid surface
 * won - the same whole-or-nothing post-pass the CPU entry applies, so the
 * caller fills its arrays from this dispatch alone. The kernel re-runs the
 * CPU's chains op-for-op (terrain splines, cave/noodle carving, the
 * banded surface search, and the full aquifer stack over the packed AN_*
 * noises and the per-seed aquifer hash seeds passed as runtime arguments)
 * in single precision, so the planes drift budgeted against the CPU path.
 * There is no failMask: a sparse chunk is rejected
 * whole-or-nothing, and rejection fires before ANY GPU work or caller-
 * output write. Returns 0 when the GPU filled all four planes for every
 * anchor; nonzero means the whole chunk was refused and the
 * caller runs the CPU path per anchor (NULL pointers, n outside [1, ~1M],
 * a BiomeNoise not fully initialized by setBiomeSeed() or pre-1.18, mode
 * off, disabled/dead engine, table-pack failure, slab/
 * allocation failure, or an enqueue failure).
 */
int gpuCfxSurfaceColumnsPointsFloat(int *outSolidY, int *outFluidY,
    int *outSurfaceY, int *outFlags, const BiomeNoise *bn,
    const int *blockX, const int *blockZ, int n);
#endif /* parked surface-columns GPU path */

/*
 * Engine metrics: a fixed uint64 slot array accumulated by the cfx batch
 * and sparse entries, snapshot by gpuQueryMetrics(). The layout is mirrored
 * by the mod's Java reader, so the slot indices below are API, not internals.
 * The plain counters (slots 1-8, 13) accumulate on every attempted batch;
 * the four event timers (slots 9-12) stay zero unless the process opts in
 * with CUBIOMES_GPU_PROFILING=1, which creates the OpenCL queue with
 * CL_QUEUE_PROFILING_ENABLE so device timestamps become available. A
 * loader/driver stack that accepts the flag and then drops it degrades the
 * timers back to zero instead of reporting garbage.
 */
#define GPU_METRICS_SLOTS 68
#define GPU_METRICS_PROFILING 0     /* 1 when the queue really profiles events */
#define GPU_METRICS_BATCH_MEMBERS 1 /* sum of n over attempted batch entries */
#define GPU_METRICS_BATCH_POINTS 2  /* sum of per-block w*h over them */
#define GPU_METRICS_KERNEL_LAUNCHES 3   /* successful kernel enqueues */
#define GPU_METRICS_H2D_WRITES 4        /* successful input-buffer writes */
#define GPU_METRICS_D2H_READS 5         /* successful readback enqueues */
#define GPU_METRICS_MUTEX_WAIT_NANOS 6  /* blocked acquiring the engine lock */
#define GPU_METRICS_HOST_SUBMIT_NANOS 7 /* lock acquired -> last enqueue */
#define GPU_METRICS_HOST_WAIT_NANOS 8   /* readback wait wall */
#define GPU_METRICS_H2D_NANOS 9     /* write event END-START sums */
#define GPU_METRICS_KERNEL_NANOS 10 /* kernel event END-START sums */
#define GPU_METRICS_D2H_NANOS 11    /* read event END-START sums */
#define GPU_METRICS_QUEUE_DELAY_NANOS 12 /* event SUBMIT-QUEUED sums */
/* Slot 13 is shim-owned (confluxnative.c): the JNI array-marshal wall of
 * the two batch entries, which the engine never sees. */
#define GPU_METRICS_ARRAY_COPY_NANOS 13
/* Slot 14 + min(n, 32) counts batch entries with n blocks (33 slots, the
 * engine's batch-entry cap). */
#define GPU_METRICS_MEMBER_HIST 14
/* Slot 47 + k counts batch entries whose total points sit in (2^(k-1),
 * 2^k] - slot 0 is a single point, the last slot (k = 20) is the dispatch
 * cap (21 slots). */
#define GPU_METRICS_POINT_HIST 47

/* Copies the current metrics into caller storage of GPU_METRICS_SLOTS
 * uint64. Measurement aid, like the dispatch counters above. */
void gpuQueryMetrics(uint64_t *out);

#endif
