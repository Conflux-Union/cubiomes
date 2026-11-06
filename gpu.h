#ifndef GPU_H_
#define GPU_H_

/*
 * Optional OpenCL GPU execution for the 1.18+ Overworld climate biome
 * sampling. The GPU engine mirrors mapOverworldSurfaceBiome() + depth-aware
 * sampleBiomeNoise() semantics exactly (the depth channel included); a parity
 * harness (gpu_test.c, `make gpu-test`) gates every change as bit-exact
 * against the CPU reference. The engine is a pure executor: it never applies
 * an approximation, and any CPU/GPU mismatch is a defect to fix, never a
 * tolerance to accept.
 *
 * Tables are packed from the live Generator/BiomeNoise structs (perm tables,
 * per-octave noise parameters, spline trees, biome tree), so all 1.18+
 * versions come free without per-version hardcoded tables in this layer.
 *
 * Thread model: one process-global engine with ONE command queue. A mutex
 * serializes submit+read around each dispatch, so concurrent callers queue up
 * but never race. Consumers that link this module need the dynamic loader
 * (-ldl on older glibc) and threads (-lpthread on older glibc).
 *
 * Mode handling: the environment variable CUBIOMES_GPU (off|auto|force) is
 * parsed at first use; gpuSetMode() overrides it. "auto" (default) uses the
 * GPU when available and silently falls back to the CPU otherwise; "off"
 * never touches the GPU; "force" is a debug aid that reports loudly when the
 * GPU is unavailable. Read-only queries (gpuGetMode/gpuAvailable) never fail
 * because of the GPU; the dispatch below returns nonzero on a GPU error and
 * latches the engine dead for the process so later calls fall back too.
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
 * Cheap health check for GPU dispatch of the given generator: OpenCL with an
 * fp64 device is present and healthy, the generator is an unseeded-special
 * 1.18+ Overworld biome-noise generator (the supported version/dimension
 * combination). Scale coverage is fixed: the dispatch supports the surface
 * scales 1 (block + voronoi) and 4 (quart). Does not validate the requested
 * area; the dispatch re-checks its own arguments.
 */
int gpuAvailable(const Generator *g);

/* Name of the probed OpenCL device, or an empty string before first use /
 * when unavailable. Diagnostic aid, not part of the dispatch contract. */
const char *gpuDeviceName(void);

/*
 * GPU equivalent of mapOverworldSurfaceBiome(out, &g->bn, sha, scale, x, z,
 * w, h) - the internal dispatch the genBiomes hook (see generator.c) falls
 * back on. Returns 0 on success; nonzero when the GPU engine is disabled,
 * dead, or the arguments are unsupported (mirrors the wrapper's error
 * convention so callers fall back to the CPU path). When `np` is non-NULL it
 * receives NP_MAX climate noise parameters per point (int64_t[np? w*h :
 * ...][6]), which the CPU wrapper does not expose; values match
 * sampleBiomeNoise()'s `np` output for the same point.
 */
int gpuMapOverworldSurfaceBiome(int *out, const Generator *g, uint64_t sha,
    int scale, int x, int z, int w, int h, int64_t *np);

#endif
