#include "terrain_features.h"

#include <math.h>
#include <string.h>

typedef struct CaveMaskContext
{
    int chunkX;
    int chunkZ;
    int minY;
    int maxY;
    uint8_t *mask;
} CaveMaskContext;

enum TreeSelector
{
    SELECT_OAK,
    SELECT_BIRCH,
    SELECT_SPRUCE,
    SELECT_FOREST,
    SELECT_MOUNTAIN,
    SELECT_SAVANNA,
    SELECT_JUNGLE,
    SELECT_JUNGLE_EDGE,
    SELECT_DARK_FOREST_BROWN,
    SELECT_DARK_FOREST_RED,
};

typedef struct TreeProfile
{
    int count;
    int extraCount;
    int decoratorIndex;
    float extraChance;
    int selector;
} TreeProfile;

static int validateGenerator(const Generator *g)
{
    if (!g)
        return FEATURE_ERR_ARGUMENT;
    if (g->mc != MC_1_17_1)
        return FEATURE_ERR_VERSION;
    if (g->dim != DIM_OVERWORLD)
        return FEATURE_ERR_DIMENSION;
    return FEATURE_OK;
}

static int supportedStructureBounds(int type)
{
    switch (type)
    {
    case Desert_Pyramid:
    case Jungle_Temple:
    case Swamp_Hut:
    case Igloo:
    case Village:
    case Monument:
        return 1;
    default:
        return 0;
    }
}

int getChunkNaturalStructures(Generator *g, int chunkX, int chunkZ,
    NaturalStructure *out, size_t capacity, size_t *required)
{
    int status = validateGenerator(g);
    if (status != FEATURE_OK)
        return status;
    if (!required || (!out && capacity != 0))
        return FEATURE_ERR_ARGUMENT;

    size_t count = 0;
    int type;
    for (type = 0; type < FEATURE_NUM; type++)
    {
        StructureConfig config;
        StructureVariant variant;
        Pos start;
        int biome;

        if (!supportedStructureBounds(type) ||
            !getStructureConfig(type, g->mc, &config) ||
            config.dim != DIM_OVERWORLD)
            continue;

        int regionX = floordiv(chunkX, config.regionSize);
        int regionZ = floordiv(chunkZ, config.regionSize);
        if (!getStructurePos(type, g->mc, g->seed, regionX, regionZ, &start) ||
            start.x != chunkX * 16 || start.z != chunkZ * 16 ||
            !isViableStructurePos(type, g, start.x, start.z, 0))
            continue;

        biome = getBiomeAt(g, 4, (start.x >> 2) + 2, 0,
            (start.z >> 2) + 2);
        if (!getVariant(&variant, type, g->mc, g->seed,
                start.x, start.z, biome) ||
            variant.sx <= 0 || variant.sy <= 0 || variant.sz <= 0)
            continue;

        if (out && count < capacity)
        {
            SurfaceNoise surface;
            float height;
            int sampleBiome;
            NaturalStructure *record = out + count;

            initSurfaceNoise(&surface, DIM_OVERWORLD, g->seed);
            if (mapApproxHeight(&height, &sampleBiome, g, &surface,
                    start.x >> 2, start.z >> 2, 1, 1) != 0)
                height = 64.0f;

            record->type = type;
            record->start = start;
            record->variant = variant;
            record->bounds.min.x = start.x + variant.x;
            record->bounds.min.y = (int)floorf(height) + variant.y - 320;
            record->bounds.min.z = start.z + variant.z;
            record->bounds.max.x = record->bounds.min.x + variant.sx - 1;
            record->bounds.max.y = record->bounds.min.y + variant.sy - 1;
            record->bounds.max.z = record->bounds.min.z + variant.sz - 1;
            record->flags = FEATURE_START_EXACT |
                FEATURE_BOUNDS_APPROXIMATE | FEATURE_START_PIECE_ONLY;
        }
        count++;
    }

    *required = count;
    if (out && count > capacity)
        return FEATURE_ERR_CAPACITY;
    return FEATURE_PARTIAL;
}

static int getTreeProfile(int biome, TreeProfile *profile)
{
    *profile = (TreeProfile){0, 1, 0, 0.1f, SELECT_OAK};
    switch (biome)
    {
    case plains:
        *profile = (TreeProfile){0, 1, 1, 0.05f, SELECT_OAK};
        return 1;
    case sunflower_plains:
        *profile = (TreeProfile){0, 1, 2, 0.05f, SELECT_OAK};
        return 1;
    case forest:
    case wooded_hills:
        *profile = (TreeProfile){10, 1, 1, 0.1f, SELECT_FOREST};
        return 1;
    case flower_forest:
        *profile = (TreeProfile){6, 1, 1, 0.1f, SELECT_FOREST};
        return 1;
    case birch_forest:
    case birch_forest_hills:
    case tall_birch_forest:
    case tall_birch_hills:
        *profile = (TreeProfile){10, 1, 1, 0.1f, SELECT_BIRCH};
        return 1;
    case taiga:
    case taiga_hills:
    case taiga_mountains:
    case snowy_taiga:
    case snowy_taiga_hills:
    case snowy_taiga_mountains:
    case giant_tree_taiga:
    case giant_tree_taiga_hills:
    case giant_spruce_taiga:
    case giant_spruce_taiga_hills:
        *profile = (TreeProfile){10, 1, 1, 0.1f, SELECT_SPRUCE};
        return 1;
    case mountains:
    case gravelly_mountains:
    case modified_gravelly_mountains:
        *profile = (TreeProfile){0, 1, 0, 0.1f, SELECT_MOUNTAIN};
        return 1;
    case mountain_edge:
    case wooded_mountains:
        *profile = (TreeProfile){3, 1, 0, 0.1f, SELECT_MOUNTAIN};
        return 1;
    case savanna:
    case savanna_plateau:
        *profile = (TreeProfile){1, 1, 1, 0.1f, SELECT_SAVANNA};
        return 1;
    case shattered_savanna:
    case shattered_savanna_plateau:
        *profile = (TreeProfile){2, 1, 0, 0.1f, SELECT_SAVANNA};
        return 1;
    case jungle:
    case jungle_hills:
        *profile = (TreeProfile){50, 1, 1, 0.1f, SELECT_JUNGLE};
        return 1;
    case modified_jungle:
        *profile = (TreeProfile){50, 1, 0, 0.1f, SELECT_JUNGLE};
        return 1;
    case jungle_edge:
    case modified_jungle_edge:
        *profile = (TreeProfile){2, 1, 0, 0.1f, SELECT_JUNGLE_EDGE};
        return 1;
    case dark_forest:
        *profile = (TreeProfile){16, 0, 0, 0.0f,
            SELECT_DARK_FOREST_BROWN};
        return 1;
    case dark_forest_hills:
        *profile = (TreeProfile){16, 0, 0, 0.0f,
            SELECT_DARK_FOREST_RED};
        return 1;
    case swamp:
    case swamp_hills:
        *profile = (TreeProfile){2, 1, 0, 0.1f, SELECT_OAK};
        return 1;
    case river:
    case frozen_river:
        *profile = (TreeProfile){0, 1, 0, 0.1f, SELECT_OAK};
        return 1;
    case snowy_tundra:
    case snowy_mountains:
    case ice_spikes:
        *profile = (TreeProfile){0, 1, 0, 0.1f, SELECT_SPRUCE};
        return 1;
    case wooded_badlands_plateau:
    case modified_wooded_badlands_plateau:
        *profile = (TreeProfile){5, 1, 0, 0.1f, SELECT_OAK};
        return 1;
    case ocean:
    case frozen_ocean:
    case deep_ocean:
    case warm_ocean:
    case lukewarm_ocean:
    case cold_ocean:
    case deep_warm_ocean:
    case deep_lukewarm_ocean:
    case deep_cold_ocean:
    case deep_frozen_ocean:
        *profile = (TreeProfile){0, 1, 0, 0.1f, SELECT_OAK};
        return 1;
    default:
        return 0;
    }
}

static int selectTreeType(uint64_t *rng, int selector)
{
    switch (selector)
    {
    case SELECT_BIRCH:
        return TREE_BIRCH;
    case SELECT_SPRUCE:
        return TREE_SPRUCE;
    case SELECT_FOREST:
        if (nextFloat(rng) < 0.2f) return TREE_BIRCH;
        if (nextFloat(rng) < 0.1f) return TREE_OAK;
        return TREE_OAK;
    case SELECT_MOUNTAIN:
        if (nextFloat(rng) < 0.666f) return TREE_SPRUCE;
        if (nextFloat(rng) < 0.1f) return TREE_OAK;
        return TREE_OAK;
    case SELECT_SAVANNA:
        return nextFloat(rng) < 0.8f ? TREE_ACACIA : TREE_OAK;
    case SELECT_JUNGLE:
        if (nextFloat(rng) < 0.1f) return TREE_OAK;
        if (nextFloat(rng) < 0.5f) return TREE_JUNGLE_BUSH;
        if (nextFloat(rng) < 0.33333334f) return TREE_JUNGLE;
        return TREE_JUNGLE;
    case SELECT_JUNGLE_EDGE:
        if (nextFloat(rng) < 0.1f) return TREE_OAK;
        if (nextFloat(rng) < 0.5f) return TREE_JUNGLE_BUSH;
        return TREE_JUNGLE;
    case SELECT_DARK_FOREST_BROWN:
        if (nextFloat(rng) < 0.025f) return TREE_HUGE_BROWN_MUSHROOM;
        if (nextFloat(rng) < 0.05f) return TREE_HUGE_RED_MUSHROOM;
        if (nextFloat(rng) < 0.6666667f) return TREE_DARK_OAK;
        if (nextFloat(rng) < 0.2f) return TREE_BIRCH;
        if (nextFloat(rng) < 0.1f) return TREE_OAK;
        return TREE_OAK;
    case SELECT_DARK_FOREST_RED:
        if (nextFloat(rng) < 0.025f) return TREE_HUGE_RED_MUSHROOM;
        if (nextFloat(rng) < 0.05f) return TREE_HUGE_BROWN_MUSHROOM;
        if (nextFloat(rng) < 0.6666667f) return TREE_DARK_OAK;
        if (nextFloat(rng) < 0.2f) return TREE_BIRCH;
        if (nextFloat(rng) < 0.1f) return TREE_OAK;
        return TREE_OAK;
    case SELECT_OAK:
    default:
        return TREE_OAK;
    }
}

static uint64_t getPopulationSeed117(uint64_t worldSeed, int chunkX, int chunkZ)
{
    uint64_t rng;
    uint64_t xMultiplier;
    uint64_t zMultiplier;
    setSeed(&rng, worldSeed);
    xMultiplier = nextLong(&rng) | 1;
    zMultiplier = nextLong(&rng) | 1;
    return ((uint64_t)(int64_t)chunkX * 16 * xMultiplier +
        (uint64_t)(int64_t)chunkZ * 16 * zMultiplier) ^ worldSeed;
}

int getChunkNaturalTreeCandidates(const Generator *g, int chunkX, int chunkZ,
    NaturalTreeCandidate *out, size_t capacity, size_t *required)
{
    int status = validateGenerator(g);
    if (status != FEATURE_OK)
        return status;
    if (!required || (!out && capacity != 0))
        return FEATURE_ERR_ARGUMENT;

    int biome = getBiomeAt(g, 4, chunkX * 4 + 2, 0, chunkZ * 4 + 2);
    TreeProfile profile;
    if (!getTreeProfile(biome, &profile))
    {
        *required = 0;
        return FEATURE_PARTIAL;
    }

    uint64_t decoratorSeed = getPopulationSeed117(g->seed, chunkX, chunkZ) +
        profile.decoratorIndex + 80000;
    uint64_t rng;
    setSeed(&rng, decoratorSeed);
    int count = profile.count;
    int darkOakGrid = profile.selector == SELECT_DARK_FOREST_BROWN ||
        profile.selector == SELECT_DARK_FOREST_RED;
    if (!darkOakGrid &&
        nextFloat(&rng) < profile.extraChance)
        count += profile.extraCount;
    *required = (size_t)count;

    if (out)
    {
        SurfaceNoise surface;
        float heights[16];
        int heightBiomes[16];
        initSurfaceNoise(&surface, DIM_OVERWORLD, g->seed);
        if (mapApproxHeight(heights, heightBiomes, g, &surface,
                chunkX * 4, chunkZ * 4, 4, 4) != 0)
        {
            int j;
            for (j = 0; j < 16; j++)
                heights[j] = 64.0f;
        }
        int i;
        for (i = 0; i < count && (size_t)i < capacity; i++)
        {
            NaturalTreeCandidate *record = out + i;
            if (darkOakGrid)
            {
                record->pos.x = chunkX * 16 + (i / 4) * 4 + 1 +
                    nextInt(&rng, 3);
                record->pos.z = chunkZ * 16 + (i % 4) * 4 + 1 +
                    nextInt(&rng, 3);
            }
            else
            {
                record->pos.x = chunkX * 16 + nextInt(&rng, 16);
                record->pos.z = chunkZ * 16 + nextInt(&rng, 16);
            }
            record->type = selectTreeType(&rng, profile.selector);
            record->biome = biome;

            int localX = record->pos.x - chunkX * 16;
            int localZ = record->pos.z - chunkZ * 16;
            record->pos.y = (int)floorf(heights[(localZ >> 2) * 4 +
                (localX >> 2)]) + 1;
            record->flags = FEATURE_TREE_CANDIDATE | FEATURE_Y_APPROXIMATE;
            if (i == 0)
                record->flags |= FEATURE_XZ_EXACT | FEATURE_TYPE_EXACT;
            else
                record->flags |= FEATURE_SEQUENCE_APPROXIMATE;
        }
    }

    if (out && (size_t)count > capacity)
        return FEATURE_ERR_CAPACITY;
    return FEATURE_OK;
}

size_t getCaveCarvingMaskSize(int minY, int maxY)
{
    if (minY < 0 || maxY > 255 || minY > maxY)
        return 0;
    return ((size_t)(maxY - minY + 1) * 256 + 7) >> 3;
}

static float mcSin(float value)
{
    int index = (int)(value * 10430.378f) & 65535;
    return (float)sin(index * 6.28318530717958647692 / 65536.0);
}

static float mcCos(float value)
{
    int index = (int)(value * 10430.378f + 16384.0f) & 65535;
    return (float)sin(index * 6.28318530717958647692 / 65536.0);
}

static void setMaskBit(CaveMaskContext *ctx, int x, int y, int z)
{
    if (y < ctx->minY || y > ctx->maxY)
        return;
    size_t bit = (size_t)(y - ctx->minY) * 256 + z * 16 + x;
    ctx->mask[bit >> 3] |= (uint8_t)(1U << (bit & 7));
}

static int canCarveBranch(const CaveMaskContext *ctx, double x, double z,
    int branchIndex, int branchCount, float baseWidth)
{
    double dx = x - (ctx->chunkX * 16 + 8);
    double dz = z - (ctx->chunkZ * 16 + 8);
    double remaining = branchCount - branchIndex;
    double reach = baseWidth + 18.0f;
    return dx * dx + dz * dz - remaining * remaining <= reach * reach;
}

static void carveRegion(CaveMaskContext *ctx, double x, double y, double z,
    double horizontalScale, double verticalScale, double floorLevel,
    const float *ravineStretch)
{
    double centerX = ctx->chunkX * 16 + 8;
    double centerZ = ctx->chunkZ * 16 + 8;
    double reach = 16.0 + horizontalScale * 2.0;
    if (fabs(x - centerX) > reach || fabs(z - centerZ) > reach)
        return;

    int startX = ctx->chunkX * 16;
    int startZ = ctx->chunkZ * 16;
    int minX = (int)floor(x - horizontalScale) - startX - 1;
    int maxX = (int)floor(x + horizontalScale) - startX;
    int minY = (int)floor(y - verticalScale) - 1;
    int maxY = (int)floor(y + verticalScale) + 1;
    int minZ = (int)floor(z - horizontalScale) - startZ - 1;
    int maxZ = (int)floor(z + horizontalScale) - startZ;

    if (minX < 0) minX = 0;
    if (maxX > 15) maxX = 15;
    if (minY < 1) minY = 1;
    if (maxY > 248) maxY = 248;
    if (minZ < 0) minZ = 0;
    if (maxZ > 15) maxZ = 15;

    int localX, localZ, blockY;
    for (localX = minX; localX <= maxX; localX++)
    {
        double scaledX = (startX + localX + 0.5 - x) / horizontalScale;
        for (localZ = minZ; localZ <= maxZ; localZ++)
        {
            double scaledZ = (startZ + localZ + 0.5 - z) / horizontalScale;
            if (scaledX * scaledX + scaledZ * scaledZ >= 1.0)
                continue;

            for (blockY = maxY; blockY > minY; blockY--)
            {
                double scaledY = (blockY - 0.5 - y) / verticalScale;
                int excluded;
                if (ravineStretch)
                {
                    excluded = (scaledX * scaledX + scaledZ * scaledZ) *
                        ravineStretch[blockY - 1] + scaledY * scaledY / 6.0 >= 1.0;
                }
                else
                {
                    excluded = scaledY <= floorLevel ||
                        scaledX * scaledX + scaledY * scaledY +
                        scaledZ * scaledZ >= 1.0;
                }
                if (!excluded)
                    setMaskBit(ctx, localX, blockY, localZ);
            }
        }
    }
}

static float getTunnelWidth(uint64_t *rng)
{
    float width = nextFloat(rng) * 2.0f + nextFloat(rng);
    if (nextInt(rng, 10) == 0)
        width *= nextFloat(rng) * nextFloat(rng) * 3.0f + 1.0f;
    return width;
}

static void carveTunnels(CaveMaskContext *ctx, uint64_t seed,
    double x, double y, double z, double horizontalScale,
    double verticalScale, float width, float yaw, float pitch,
    int branchStart, int branchCount, double yawPitchRatio,
    double floorLevel)
{
    uint64_t rng;
    setSeed(&rng, seed);
    int split = nextInt(&rng, branchCount / 2) + branchCount / 4;
    int steep = nextInt(&rng, 6) == 0;
    float yawVelocity = 0.0f;
    float pitchVelocity = 0.0f;
    int branch;

    for (branch = branchStart; branch < branchCount; branch++)
    {
        double radius = 1.5 + mcSin((float)3.14159265358979323846 *
            branch / branchCount) * width;
        double verticalRadius = radius * yawPitchRatio;
        float cosPitch = mcCos(pitch);
        x += mcCos(yaw) * cosPitch;
        y += mcSin(pitch);
        z += mcSin(yaw) * cosPitch;
        pitch *= steep ? 0.92f : 0.7f;
        pitch += pitchVelocity * 0.1f;
        yaw += yawVelocity * 0.1f;
        pitchVelocity *= 0.9f;
        yawVelocity *= 0.75f;
        pitchVelocity += (nextFloat(&rng) - nextFloat(&rng)) *
            nextFloat(&rng) * 2.0f;
        yawVelocity += (nextFloat(&rng) - nextFloat(&rng)) *
            nextFloat(&rng) * 4.0f;

        if (branch == split && width > 1.0f)
        {
            uint64_t leftSeed = nextLong(&rng);
            float leftWidth = nextFloat(&rng) * 0.5f + 0.5f;
            carveTunnels(ctx, leftSeed, x, y, z, horizontalScale,
                verticalScale, leftWidth,
                yaw - (float)(3.14159265358979323846 / 2.0),
                pitch / 3.0f, branch, branchCount, 1.0, floorLevel);

            uint64_t rightSeed = nextLong(&rng);
            float rightWidth = nextFloat(&rng) * 0.5f + 0.5f;
            carveTunnels(ctx, rightSeed, x, y, z, horizontalScale,
                verticalScale, rightWidth,
                yaw + (float)(3.14159265358979323846 / 2.0),
                pitch / 3.0f, branch, branchCount, 1.0, floorLevel);
            return;
        }

        if (nextInt(&rng, 4) != 0)
        {
            if (!canCarveBranch(ctx, x, z, branch, branchCount, width))
                return;
            carveRegion(ctx, x, y, z, radius * horizontalScale,
                verticalRadius * verticalScale, floorLevel, NULL);
        }
    }
}

static int biasedHeight(uint64_t *rng, int minY, int maxY, int inner)
{
    int outer = nextInt(rng, maxY - minY - inner + 1);
    return nextInt(rng, outer + inner) + minY;
}

static void carveCaves(CaveMaskContext *ctx, uint64_t *rng,
    int sourceChunkX, int sourceChunkZ)
{
    int systems = nextInt(rng, nextInt(rng, nextInt(rng, 15) + 1) + 1);
    int system;
    for (system = 0; system < systems; system++)
    {
        double x = sourceChunkX * 16 + nextInt(rng, 16);
        double y = biasedHeight(rng, 0, 127, 8);
        double z = sourceChunkZ * 16 + nextInt(rng, 16);
        int tunnels = 1;

        if (nextInt(rng, 4) == 0)
        {
            float radius = 1.0f + nextFloat(rng) * 6.0f;
            nextLong(rng);
            carveRegion(ctx, x + 1.0, y, z, 1.5 + radius,
                (1.5 + radius) * 0.5, -0.7, NULL);
            tunnels += nextInt(rng, 4);
        }

        int tunnel;
        for (tunnel = 0; tunnel < tunnels; tunnel++)
        {
            float yaw = nextFloat(rng) * (float)(3.14159265358979323846 * 2.0);
            float pitch = (nextFloat(rng) - 0.5f) / 4.0f;
            float width = getTunnelWidth(rng);
            int branchCount = 112 - nextInt(rng, 28);
            uint64_t tunnelSeed = nextLong(rng);
            carveTunnels(ctx, tunnelSeed, x, y, z, 1.0, 1.0,
                width, yaw, pitch, 0, branchCount, 1.0, -0.7);
        }
    }
}

static void carveRavine(CaveMaskContext *ctx, uint64_t seed,
    double x, double y, double z, float width, float yaw, float pitch,
    int branchCount)
{
    uint64_t rng;
    float stretch[256];
    float factor = 1.0f;
    int i;

    setSeed(&rng, seed);
    for (i = 0; i < 256; i++)
    {
        if (i == 0 || nextInt(&rng, 3) == 0)
            factor = 1.0f + nextFloat(&rng) * nextFloat(&rng);
        stretch[i] = factor * factor;
    }

    float yawVelocity = 0.0f;
    float pitchVelocity = 0.0f;
    for (i = 0; i < branchCount; i++)
    {
        double radius = 1.5 + mcSin(i * (float)3.14159265358979323846 /
            branchCount) * width;
        double verticalRadius = radius * 3.0;
        radius *= 0.75f + nextFloat(&rng) * 0.25f;
        verticalRadius *= 0.75f + nextFloat(&rng) * 0.25f;

        float cosPitch = mcCos(pitch);
        x += mcCos(yaw) * cosPitch;
        y += mcSin(pitch);
        z += mcSin(yaw) * cosPitch;
        pitch *= 0.7f;
        pitch += pitchVelocity * 0.05f;
        yaw += yawVelocity * 0.05f;
        pitchVelocity *= 0.8f;
        yawVelocity *= 0.5f;
        pitchVelocity += (nextFloat(&rng) - nextFloat(&rng)) *
            nextFloat(&rng) * 2.0f;
        yawVelocity += (nextFloat(&rng) - nextFloat(&rng)) *
            nextFloat(&rng) * 4.0f;

        if (nextInt(&rng, 4) != 0)
        {
            if (!canCarveBranch(ctx, x, z, i, branchCount, width))
                return;
            carveRegion(ctx, x, y, z, radius, verticalRadius, 0.0,
                stretch);
        }
    }
}

static void runSourceCarvers(const Generator *g, CaveMaskContext *ctx,
    int sourceChunkX, int sourceChunkZ)
{
    int biome = getBiomeAt(g, 4, sourceChunkX * 4, 0, sourceChunkZ * 4);
    float caveProbability = isOceanic(biome) ? 0.06666667f : 0.14285715f;
    uint64_t base;
    uint64_t first;
    uint64_t second;
    uint64_t rng;

    setSeed(&base, g->seed);
    first = nextLong(&base);
    second = nextLong(&base);

    setSeed(&rng, (uint64_t)sourceChunkX * first ^
        (uint64_t)sourceChunkZ * second ^ g->seed);
    if (nextFloat(&rng) <= caveProbability)
        carveCaves(ctx, &rng, sourceChunkX, sourceChunkZ);

    setSeed(&base, g->seed + 1);
    first = nextLong(&base);
    second = nextLong(&base);
    setSeed(&rng, (uint64_t)sourceChunkX * first ^
        (uint64_t)sourceChunkZ * second ^ (g->seed + 1));
    if (nextFloat(&rng) <= 0.02f)
    {
        double x = sourceChunkX * 16 + nextInt(&rng, 16);
        double y = biasedHeight(&rng, 20, 67, 8);
        double z = sourceChunkZ * 16 + nextInt(&rng, 16);
        float yaw = nextFloat(&rng) * (float)(3.14159265358979323846 * 2.0);
        float pitch = -0.125f + nextFloat(&rng) * 0.25f;
        float width = nextFloat(&rng) * 4.0f + nextFloat(&rng) * 2.0f;
        float distanceFactor = 0.75f + nextFloat(&rng) * 0.25f;
        int branchCount = (int)(112 * distanceFactor);
        uint64_t ravineSeed = nextLong(&rng);
        carveRavine(ctx, ravineSeed, x, y, z, width, yaw, pitch, branchCount);
    }
}

int getChunkCaveCarvingMask(const Generator *g, int chunkX, int chunkZ,
    int minY, int maxY, uint8_t *mask, size_t maskSize)
{
    int status = validateGenerator(g);
    if (status != FEATURE_OK)
        return status;
    size_t required = getCaveCarvingMaskSize(minY, maxY);
    if (!mask || required == 0)
        return FEATURE_ERR_ARGUMENT;
    if (maskSize < required)
        return FEATURE_ERR_CAPACITY;

    memset(mask, 0, required);
    CaveMaskContext ctx = {chunkX, chunkZ, minY, maxY, mask};
    int dz, dx;
    for (dx = -8; dx <= 8; dx++)
        for (dz = -8; dz <= 8; dz++)
            runSourceCarvers(g, &ctx, chunkX + dx, chunkZ + dz);
    return FEATURE_OK;
}
