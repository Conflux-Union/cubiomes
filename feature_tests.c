#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "terrain_features.h"

static int maskGet(const uint8_t *mask, int minY, int x, int y, int z)
{
    size_t bit = (size_t)(y - minY) * 256 + z * 16 + x;
    return (mask[bit >> 3] >> (bit & 7)) & 1;
}

static uint64_t maskHash(const uint8_t *mask, size_t size)
{
    uint64_t hash = 1469598103934665603ULL;
    size_t i;
    for (i = 0; i < size; i++)
    {
        hash ^= mask[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void testUnsupportedGenerator(void)
{
    Generator g;
    NaturalStructure structure;
    size_t count = 0;

    setupGenerator(&g, MC_1_16, 0);
    applySeed(&g, DIM_OVERWORLD, 1);
    assert(getChunkNaturalStructures(&g, 0, 0, &structure, 1, &count)
        == FEATURE_ERR_VERSION);

    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_NETHER, 1);
    assert(getChunkNaturalStructures(&g, 0, 0, &structure, 1, &count)
        == FEATURE_ERR_DIMENSION);
}

static void testIglooStartBounds(void)
{
    Generator g;
    NaturalStructure structures[4];
    size_t count = 0;

    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_OVERWORLD, 6512112982729996127ULL);
    assert(getChunkNaturalStructures(&g, 16, 2, structures, 4, &count)
        == FEATURE_PARTIAL);
    assert(count == 1);
    assert(structures[0].type == Igloo);
    assert(structures[0].start.x == 256);
    assert(structures[0].start.z == 32);
    assert(structures[0].bounds.min.x == 256);
    assert(structures[0].bounds.min.z == 32);
    assert(structures[0].bounds.max.x == 262);
    assert(structures[0].bounds.max.z == 39);
    assert((structures[0].flags & FEATURE_START_EXACT) != 0);
    assert((structures[0].flags & FEATURE_BOUNDS_APPROXIMATE) != 0);
    assert((structures[0].flags & FEATURE_START_PIECE_ONLY) != 0);
}

static void testRavineDistanceFactor(void)
{
    Generator g;
    const int minY = 0;
    const int maxY = 127;
    size_t maskSize = getCaveCarvingMaskSize(minY, maxY);
    uint8_t *mask = calloc(maskSize, 1);

    assert(mask != NULL);
    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_OVERWORLD, 6512112982729996127ULL);
    assert(getChunkCaveCarvingMask(&g, -22, -32, minY, maxY,
        mask, maskSize) == FEATURE_OK);
    assert(maskHash(mask, maskSize) == 10530590817527261911ULL);
    free(mask);
}

static void testCaveCarvingMask(void)
{
    Generator g;
    const int minY = 0;
    const int maxY = 127;
    size_t maskSize = getCaveCarvingMaskSize(minY, maxY);
    uint8_t *mask = calloc(maskSize, 1);

    assert(mask != NULL);
    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_OVERWORLD, 6512112982729996127ULL);
    assert(getChunkCaveCarvingMask(&g, 1, 6, minY, maxY, mask, maskSize)
        == FEATURE_OK);
    assert(maskHash(mask, maskSize) == 9529250606001910822ULL);
    assert(maskGet(mask, minY, 7, 2, 0));
    assert(!maskGet(mask, minY, 15, 60, 15));
    free(mask);
}

static void testTreeCandidates(void)
{
    Generator g;
    NaturalTreeCandidate first[64];
    NaturalTreeCandidate second[64];
    size_t firstCount = 0;
    size_t secondCount = 0;

    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_OVERWORLD, 6512112982729996127ULL);
    assert(getChunkNaturalTreeCandidates(&g, 16, 0, first, 64,
        &firstCount) == FEATURE_OK);
    assert(getChunkNaturalTreeCandidates(&g, 16, 0, second, 64,
        &secondCount) == FEATURE_OK);
    assert(firstCount > 0 && firstCount == secondCount);
    assert(memcmp(first, second, firstCount * sizeof(*first)) == 0);
    assert(firstCount == 10);
    assert(first[0].pos.x == 258);
    assert(first[0].pos.z == 4);
    assert(first[0].type == TREE_SPRUCE);
    assert(first[0].biome == snowy_taiga);
    assert(first[0].pos.x >= 16 * 16 && first[0].pos.x < 17 * 16);
    assert(first[0].pos.z >= 0 && first[0].pos.z < 16);
    assert((first[0].flags & FEATURE_TREE_CANDIDATE) != 0);
    assert((first[0].flags & FEATURE_XZ_EXACT) != 0);
    assert((first[0].flags & FEATURE_TYPE_EXACT) != 0);
    assert((first[0].flags & FEATURE_Y_APPROXIMATE) != 0);
    if (firstCount > 1)
        assert((first[1].flags & FEATURE_SEQUENCE_APPROXIMATE) != 0);

    size_t required = 0;
    assert(getChunkNaturalTreeCandidates(&g, 16, 0, NULL, 0,
        &required) == FEATURE_OK);
    assert(required == firstCount);
    assert(getChunkNaturalTreeCandidates(&g, 16, 0, first, 1,
        &required) == FEATURE_ERR_CAPACITY);
    assert(required == firstCount);
}

static void testJungleBushCandidate(void)
{
    Generator g;
    NaturalTreeCandidate trees[64];
    size_t count = 0;

    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_OVERWORLD, 6512112982729996127ULL);
    assert(getChunkNaturalTreeCandidates(&g, 59, -318, trees, 64,
        &count) == FEATURE_OK);
    assert(count == 2);
    assert(trees[0].biome == jungle_edge);
    assert(trees[0].pos.x == 957);
    assert(trees[0].pos.z == -5075);
    assert(trees[0].type == TREE_JUNGLE_BUSH);
    assert((trees[0].flags & FEATURE_TYPE_EXACT) != 0);
}

static void testDarkForestCandidates(void)
{
    Generator g;
    NaturalTreeCandidate trees[16];
    size_t count = 0;

    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_OVERWORLD, 6512112982729996127ULL);
    assert(getChunkNaturalTreeCandidates(&g, -490, -512, trees, 16,
        &count) == FEATURE_OK);
    assert(count == 16);
    assert(trees[0].biome == dark_forest);
    assert(trees[0].pos.x == -7839);
    assert(trees[0].pos.z == -8191);
    assert(trees[0].type == TREE_DARK_OAK);
    assert((trees[0].flags & FEATURE_XZ_EXACT) != 0);
    assert((trees[0].flags & FEATURE_TYPE_EXACT) != 0);
}

static void testUnsupportedVegetationPipeline(void)
{
    Generator g;
    size_t count = 1;

    setupGenerator(&g, MC_1_17_1, 0);
    applySeed(&g, DIM_OVERWORLD, 6512112982729996127ULL);
    assert(getChunkNaturalTreeCandidates(&g, 59, -316, NULL, 0,
        &count) == FEATURE_PARTIAL);
    assert(count == 0);
}

int main(void)
{
    testUnsupportedGenerator();
    testIglooStartBounds();
    testCaveCarvingMask();
    testRavineDistanceFactor();
    testTreeCandidates();
    testJungleBushCandidate();
    testDarkForestCandidates();
    testUnsupportedVegetationPipeline();
    return 0;
}
