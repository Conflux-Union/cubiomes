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
 * serializes submit+read around each dispatch, so concurrent callers queue up
 * but never race. Dispatches hold the mutex through kernel execution and a
 * blocking readback: a wedged GPU hangs the calling thread (the promise is
 * that GPU calls never fail or change results, never that they cannot hang).
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

#endif
