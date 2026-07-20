#ifndef TERRAIN_FEATURES_H_
#define TERRAIN_FEATURES_H_

#include <stddef.h>
#include <stdint.h>

#include "finders.h"

#ifdef __cplusplus
extern "C"
{
#endif

enum FeatureStatus
{
    FEATURE_OK = 0,
    /** Results are valid, but the query does not cover every vanilla case. */
    FEATURE_PARTIAL = 1,
    FEATURE_ERR_ARGUMENT = -1,
    FEATURE_ERR_VERSION = -2,
    FEATURE_ERR_DIMENSION = -3,
    FEATURE_ERR_CAPACITY = -4,
};

enum FeatureAccuracy
{
    FEATURE_START_EXACT = 1 << 0,
    FEATURE_BOUNDS_APPROXIMATE = 1 << 1,
    FEATURE_START_PIECE_ONLY = 1 << 2,
    FEATURE_CARVER_GEOMETRY = 1 << 3,
    FEATURE_TREE_CANDIDATE = 1 << 4,
    FEATURE_XZ_EXACT = 1 << 5,
    FEATURE_TYPE_EXACT = 1 << 6,
    FEATURE_Y_APPROXIMATE = 1 << 7,
    FEATURE_SEQUENCE_APPROXIMATE = 1 << 8,
};

enum NaturalTreeType
{
    TREE_OAK,
    TREE_BIRCH,
    TREE_SPRUCE,
    TREE_JUNGLE,
    TREE_ACACIA,
    TREE_DARK_OAK,
    TREE_JUNGLE_BUSH,
    TREE_HUGE_BROWN_MUSHROOM,
    TREE_HUGE_RED_MUSHROOM,
};

STRUCT(FeatureBox)
{
    Pos3 min;
    Pos3 max;
};

/**
 * A structure start in a Minecraft 1.17.1 chunk.
 *
 * The start coordinate and structure type are exact. The bounds describe the
 * starting template only. Their horizontal placement and Y coordinate are an
 * estimate because cubiomes does not generate template pools or block terrain.
 */
STRUCT(NaturalStructure)
{
    int type;
    Pos start;
    FeatureBox bounds;
    uint32_t flags;
    StructureVariant variant;
};

/**
 * A tree-like vegetation placement candidate produced by the 1.17.1 biome
 * decorator. Jungle bushes and huge mushrooms have distinct type values so
 * callers can exclude them when they only need trees.
 *
 * A candidate can still fail to become a tree when vanilla checks the ground,
 * available volume, fluids, and neighbouring features. The first candidate's
 * X/Z and type follow the vanilla random stream exactly. Later candidates are
 * estimates because generating the preceding feature consumes terrain-
 * dependent random values. Y always uses cubiomes' approximate height map.
 */
STRUCT(NaturalTreeCandidate)
{
    Pos3 pos;
    int type;
    int biome;
    uint32_t flags;
};

/**
 * Finds supported natural structure starts whose start chunk is (chunkX,
 * chunkZ). Bounds are currently available for desert pyramids, jungle
 * temples, swamp huts, igloos, villages, and monuments. Successful queries
 * return FEATURE_PARTIAL because other 1.17.1 structure types are not covered.
 * Passing out=NULL and capacity=0 performs a count-only query. required
 * receives the number of records needed in either mode.
 */
int getChunkNaturalStructures(Generator *g, int chunkX, int chunkZ,
    NaturalStructure *out, size_t capacity, size_t *required);

/**
 * Gets tree-like vegetation placement candidates whose origin is in
 * (chunkX,chunkZ). Passing out=NULL and capacity=0 performs a count-only query.
 * Biomes with unsupported decoration pipelines return FEATURE_PARTIAL and
 * zero records.
 */
int getChunkNaturalTreeCandidates(const Generator *g, int chunkX, int chunkZ,
    NaturalTreeCandidate *out, size_t capacity, size_t *required);

/**
 * Returns the bytes required for a 16x16 carving mask covering [minY,maxY].
 * Returns zero for an invalid interval or Y values outside 1.17.1 build limits.
 */
size_t getCaveCarvingMaskSize(int minY, int maxY);

/**
 * Generates seed-determined AIR carver candidates for a 1.17.1 Overworld
 * chunk. A set bit at ((y-minY)*256 + z*16 + x) means a cave or canyon carver
 * selected that block geometrically. This is not the final vanilla carving
 * mask or a cave_air block map: fluid vetoes, uncarvable materials, lava below
 * Y=11, and later world-generation changes require block terrain and are
 * intentionally absent.
 */
int getChunkCaveCarvingMask(const Generator *g, int chunkX, int chunkZ,
    int minY, int maxY, uint8_t *mask, size_t maskSize);

#ifdef __cplusplus
}
#endif

#endif /* TERRAIN_FEATURES_H_ */
