#include "biomenoise.h"

#include "tables/btree18.h"
#include "tables/btree192.h"
#include "tables/btree19.h"
#include "tables/btree20.h"
#include "tables/btree21wd.h"
#include "tables/btree215.h"
#include "tables/btree262.h"
#include "tables/btree263.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>


//==============================================================================
// Noise
//==============================================================================


void initSurfaceNoise(SurfaceNoise *sn, int dim, uint64_t seed)
{
    uint64_t s;
    setSeed(&s, seed);
    octaveInit(&sn->octmin, &s, sn->oct+0, -15, 16);
    octaveInit(&sn->octmax, &s, sn->oct+16, -15, 16);
    octaveInit(&sn->octmain, &s, sn->oct+32, -7, 8);
    if (dim == DIM_END)
    {
        sn->xzScale = 2.0;
        sn->yScale = 1.0;
        sn->xzFactor = 80;
        sn->yFactor = 160;
    }
    else // DIM_OVERWORLD
    {
        octaveInit(&sn->octsurf, &s, sn->oct+40, -3, 4);
        skipNextN(&s, 262*10);
        octaveInit(&sn->octdepth, &s, sn->oct+44, -15, 16);
        sn->xzScale = 0.9999999814507745;
        sn->yScale = 0.9999999814507745;
        sn->xzFactor = 80;
        sn->yFactor = 160;
    }
}

void initSurfaceNoiseBeta(SurfaceNoiseBeta *snb, uint64_t seed)
{
    uint64_t s;
    setSeed(&s, seed);

    octaveInitBeta(&snb->octmin, &s, snb->oct+0, 16, 684.412, 0.5, 1.0, 2.0);
    octaveInitBeta(&snb->octmax, &s, snb->oct+16, 16, 684.412, 0.5, 1.0, 2.0);
    octaveInitBeta(&snb->octmain, &s, snb->oct+32, 8, 684.412/80.0, 0.5, 1.0, 2.0);
    skipNextN(&s, 262*8);
    octaveInitBeta(&snb->octcontA, &s, snb->oct+40, 10, 1.121, 0.5, 1.0, 2.0);
    octaveInitBeta(&snb->octcontB, &s, snb->oct+50, 16, 200.0, 0.5, 1.0, 2.0);
}

double sampleSurfaceNoiseBetween(const SurfaceNoise *sn, int x, int y, int z,
    double noiseMin, double noiseMax)
{
    double persist, amp;
    double dx, dy, dz, sy;
    int i;

    double xzScale = 684.412 * sn->xzScale;
    double yScale = 684.412 * sn->yScale;
    double vmin = 0;
    double vmax = 0;

    persist = 1.0 / 32768.0;
    amp = 64.0;

    for (i = 15; i >= 0; i--)
    {
        dx = x * xzScale * persist;
        dz = z * xzScale * persist;
        sy = yScale * persist;
        dy = y * sy;

        vmin += samplePerlin(&sn->octmin.octaves[i], dx, dy, dz, sy, dy) * amp;
        vmax += samplePerlin(&sn->octmax.octaves[i], dx, dy, dz, sy, dy) * amp;
        if (vmin - amp > noiseMax && vmax - amp > noiseMax)
            return noiseMax;
        if (vmin + amp < noiseMin && vmax + amp < noiseMin)
            return noiseMin;

        amp *= 0.5;
        persist *= 2.0;
    }

    double xzStep = xzScale / sn->xzFactor;
    double yStep = yScale / sn->yFactor;
    double vmain = 0.5;

    persist = 1.0 / 128.0;
    amp = 0.05 * 128.0;

    for (i = 7; i >= 0; i--)
    {
        dx = x * xzStep * persist;
        dz = z * xzStep * persist;
        sy = yStep * persist;
        dy = y * sy;

        vmain += samplePerlin(&sn->octmain.octaves[i], dx, dy, dz, sy, dy) * amp;
        if (vmain - amp > 1) return vmax;
        if (vmain + amp < 0) return vmin;

        amp *= 0.5;
        persist *= 2.0;
    }

    return clampedLerp(vmain, vmin, vmax);
}

double sampleSurfaceNoise(const SurfaceNoise *sn, int x, int y, int z)
{
    double xzScale = 684.412 * sn->xzScale;
    double yScale = 684.412 * sn->yScale;
    double xzStep = xzScale / sn->xzFactor;
    double yStep = yScale / sn->yFactor;

    double minNoise = 0;
    double maxNoise = 0;
    double mainNoise = 0;
    double persist = 1.0;
    double contrib = 1.0;
    double dx, dy, dz, sy, ty;
    int i;

    for (i = 0; i < 16; i++)
    {
        dx = maintainPrecision(x * xzScale * persist);
        dy = maintainPrecision(y * yScale  * persist);
        dz = maintainPrecision(z * xzScale * persist);
        sy = yScale * persist;
        ty = y * sy;

        minNoise += samplePerlin(&sn->octmin.octaves[i], dx, dy, dz, sy, ty) * contrib;
        maxNoise += samplePerlin(&sn->octmax.octaves[i], dx, dy, dz, sy, ty) * contrib;

        if (i < 8)
        {
            dx = maintainPrecision(x * xzStep * persist);
            dy = maintainPrecision(y * yStep  * persist);
            dz = maintainPrecision(z * xzStep * persist);
            sy = yStep * persist;
            ty = y * sy;
            mainNoise += samplePerlin(&sn->octmain.octaves[i], dx, dy, dz, sy, ty) * contrib;
        }
        persist *= 0.5;
        contrib *= 2.0;
    }

    return clampedLerp(0.5 + 0.05*mainNoise, minNoise/512.0, maxNoise/512.0);
}

//==============================================================================
// Nether (1.16+) and End (1.9+) Biome Generation
//==============================================================================

void setNetherSeed(NetherNoise *nn, uint64_t seed)
{
    uint64_t s;
    setSeed(&s, seed);
    doublePerlinInit(&nn->temperature, &s, &nn->oct[0], &nn->oct[2], -7, 2);
    setSeed(&s, seed+1);
    doublePerlinInit(&nn->humidity, &s, &nn->oct[4], &nn->oct[6], -7, 2);
}

/* Gets the 3D nether biome at scale 1:4 (for 1.16+).
 */
int getNetherBiome(const NetherNoise *nn, int x, int y, int z, float *ndel)
{
    const float npoints[5][4] = {
        { 0,    0,      0,              nether_wastes       },
        { 0,   -0.5,    0,              soul_sand_valley    },
        { 0.4,  0,      0,              crimson_forest      },
        { 0,    0.5,    0.375*0.375,    warped_forest       },
        {-0.5,  0,      0.175*0.175,    basalt_deltas       },
    };

    y = 0;
    float temp = sampleDoublePerlin(&nn->temperature, x, y, z);
    float humidity = sampleDoublePerlin(&nn->humidity, x, y, z);

    int i, id = 0;
    float dmin = FLT_MAX;
    float dmin2 = FLT_MAX;
    for (i = 0; i < 5; i++)
    {
        float dx = npoints[i][0] - temp;
        float dy = npoints[i][1] - humidity;
        float dsq = dx*dx + dy*dy + npoints[i][2];
        if (dsq < dmin)
        {
            dmin2 = dmin;
            dmin = dsq;
            id = i;
        }
        else if (dsq < dmin2)
            dmin2 = dsq;
    }

    if (ndel)
        *ndel = sqrtf(dmin2) - sqrtf(dmin);

    id = (int) npoints[id][3];
    return id;
}


static void fillRad3D(int *out, int x, int y, int z, int sx, int sy, int sz,
    int id, float rad)
{
    int r, rsq;
    int i, j, k;
    r = (int) (rad);
    if (r <= 0)
        return;
    rsq = (int) floor(rad * rad);

    for (k = -r; k <= r; k++)
    {
        int ak = y+k;
        if (ak < 0 || ak >= sy)
            continue;
        int ksq = k*k;
        int *yout = &out[(int64_t)ak*sx*sz];

        for (j = -r; j <= r; j++)
        {
            int aj = z+j;
            if (aj < 0 || aj >= sz)
                continue;
            int jksq = j*j + ksq;
            for (i = -r; i <= r; i++)
            {
                int ai = x+i;
                if (ai < 0 || ai >= sx)
                    continue;
                int ijksq = i*i + jksq;
                if (ijksq > rsq)
                    continue;

                yout[(int64_t)aj*sx+ai] = id;
            }
        }
    }
}

int mapNether3D(const NetherNoise *nn, int *out, Range r, float confidence)
{
    int64_t i, j, k;
    if (r.sy <= 0)
        r.sy = 1;
    if (r.scale <= 3)
    {
        printf("mapNether3D() invalid scale for this function\n");
        return 1;
    }
    int scale = r.scale / 4;

    memset(out, 0, sizeof(int) * r.sx*r.sy*r.sz);

    // The noisedelta is the distance between the first and second closest
    // biomes within the noise space. Dividing this by the greatest possible
    // gradient (~0.05) gives a minimum diameter of voxels around the sample
    // cell that will have the same biome.
    float invgrad = 1.0 / (confidence * 0.05 * 2) / scale;

    for (k = 0; k < r.sy; k++)
    {
        int *yout = &out[k*r.sx*r.sz];

        for (j = 0; j < r.sz; j++)
        {
            for (i = 0; i < r.sx; i++)
            {
                if (yout[j*r.sx+i])
                    continue;
                //yout[j*w+i] = getNetherBiome(nn, x+i, y+k, z+j, NULL);
                //continue;

                float noisedelta;
                int xi = (r.x+i)*scale;
                int yk = (r.y+k);
                int zj = (r.z+j)*scale;
                int v = getNetherBiome(nn, xi, yk, zj, &noisedelta);
                yout[j*r.sx+i] = v;
                float cellrad = noisedelta * invgrad;
                fillRad3D(out, i, j, k, r.sx, r.sy, r.sz, v, cellrad);
            }
        }
    }
    return 0;
}

int mapNether2D(const NetherNoise *nn, int *out, int x, int z, int w, int h)
{
    Range r = {4, x, z, w, h, 0, 1};
    return mapNether3D(nn, out, r, 1.0);
}

int genNetherScaled(const NetherNoise *nn, int *out, Range r, int mc, uint64_t sha)
{
    if (r.scale <= 0) r.scale = 4;
    if (r.sy == 0) r.sy = 1;

    uint64_t siz = (uint64_t)r.sx*r.sy*r.sz;

    if (mc <= MC_1_15)
    {
        uint64_t i;
        for (i = 0; i < siz; i++)
            out[i] = nether_wastes;
        return 0;
    }

    if (r.scale == 1)
    {
        Range s = getVoronoiSrcRange(r);
        int *src;
        if (siz > 1)
        {   // the source range is large enough that we can try optimizing
            src = out + siz;
            int err = mapNether3D(nn, src, s, 1.0);
            if (err)
                return err;
        }
        else
        {
            src = NULL;
        }

        int i, j, k;
        int *p = out;
        for (k = 0; k < r.sy; k++)
        {
            for (j = 0; j < r.sz; j++)
            {
                for (i = 0; i < r.sx; i++)
                {
                    int x4, z4, y4;
                    voronoiAccess3D(sha, r.x+i, r.y+k, r.z+j, &x4, &y4, &z4);
                    if (src)
                    {
                        x4 -= s.x; y4 -= s.y; z4 -= s.z;
                        *p = src[(int64_t)y4*s.sx*s.sz + (int64_t)z4*s.sx + x4];
                    }
                    else
                    {
                        *p = getNetherBiome(nn, x4, y4, z4, NULL);
                    }
                    p++;
                }
            }
        }
        return 0;
    }
    else
    {
        return mapNether3D(nn, out, r, 1.0);
    }
}


void setEndSeed(EndNoise *en, int mc, uint64_t seed)
{
    uint64_t s;
    setSeed(&s, seed);
    skipNextN(&s, 17292);
    perlinInit(&en->perlin, &s);
    en->mc = mc;
}

static int getEndBiome(int hx, int hz, const uint16_t *hmap, int hw)
{
    int i, j;
    const uint16_t ds[26] = { // (25-2*i)*(25-2*i)
        //  0    1    2    3    4    5    6    7    8    9   10   11   12
          625, 529, 441, 361, 289, 225, 169, 121,  81,  49,  25,   9,   1,
        // 13   14   15   16   17   18   19   20   21   22   23   24,  25
            1,   9,  25,  49,  81, 121, 169, 225, 289, 361, 441, 529, 625,
    };

    const uint16_t *p_dsi = ds + (hx < 0);
    const uint16_t *p_dsj = ds + (hz < 0);
    const uint16_t *p_elev = hmap;
    uint32_t h;

    if (abs(hx) <= 15 && abs(hz) <= 15)
        h = 64 * (hx*hx + hz*hz);
    else
        h = 14401;

    for (j = 0; j < 25; j++)
    {
        uint16_t dsj = p_dsj[j];
        uint16_t e;
        uint32_t u;

        // force unroll for(i=0;i<25;i++) in a cross compatible way
        #define x5(i,x)    { x; i++; x; i++; x; i++; x; i++; x; i++; }
        #define for25(i,x) { i = 0; x5(i,x) x5(i,x) x5(i,x) x5(i,x) x5(i,x) }
        for25(i,
            if unlikely(e = p_elev[i])
            {
                if ((u = (p_dsi[i] + (uint32_t)dsj) * e) < h)
                    h = u;
            }
        );
        #undef for25
        #undef x5
        p_elev += hw;
    }

    if (h < 3600)
        return end_highlands;
    else if (h <= 10000)
        return end_midlands;
    else if (h <= 14400)
        return end_barrens;

    return small_end_islands;
}

int mapEndBiome(const EndNoise *en, int *out, int x, int z, int w, int h)
{
    int64_t i, j;
    int64_t hw = w + 26;
    int64_t hh = h + 26;
    uint16_t *hmap = (uint16_t*) malloc(sizeof(*hmap) * hw * hh);

    for (j = 0; j < hh; j++)
    {
        for (i = 0; i < hw; i++)
        {
            int64_t rx = x + i - 12;
            int64_t rz = z + j - 12;
            uint64_t rsq = rx * rx + rz * rz;
            uint16_t v = 0;
            if (rsq > 4096 && sampleSimplex2D(&en->perlin, rx, rz) < -0.9f)
            {
                //v = (llabs(rx) * 3439 + llabs(rz) * 147) % 13 + 9;
                v = (unsigned int)(
                        fabsf((float)rx) * 3439.0f + fabsf((float)rz) * 147.0f
                    ) % 13 + 9;
                v *= v;
            }
            hmap[(int64_t)j*hw+i] = v;
        }
    }

    for (j = 0; j < h; j++)
    {
        for (i = 0; i < w; i++)
        {
            int64_t hx = (i+x);
            int64_t hz = (j+z);
            uint64_t rsq = hx * hx + hz * hz;

            if (rsq <= 4096L)
                out[j*w+i] = the_end;
            else
            {
                hx = 2*hx + 1;
                hz = 2*hz + 1;
                if (en->mc > MC_1_13)
                {   // add outer end rings
                    rsq = hx * hx + hz * hz;
                    if ((int)rsq < 0)
                    {
                        out[j*w+i] = end_barrens;
                        continue;
                    }
                }
                uint16_t *p_elev = &hmap[(hz/2-z)*hw + (hx/2-x)];
                out[j*w+i] = getEndBiome(hx, hz, p_elev, hw);
            }
        }
    }

    free(hmap);
    return 0;
}

int mapEnd(const EndNoise *en, int *out, int x, int z, int w, int h)
{
    int cx = x >> 2;
    int cz = z >> 2;
    int64_t cw = ((x+w) >> 2) + 1 - cx;
    int64_t ch = ((z+h) >> 2) + 1 - cz;

    int *buf = (int*) malloc(sizeof(int) * cw * ch);
    mapEndBiome(en, buf, cx, cz, cw, ch);

    int i, j;

    for (j = 0; j < h; j++)
    {
        int cj = ((z+j) >> 2) - cz;
        for (i = 0; i < w; i++)
        {
            int ci = ((x+i) >> 2) - cx;
            int v = buf[cj*cw+ci];
            out[j*w+i] = v;
        }
    }

    free(buf);
    return 0;
}

/* Samples the End height. The coordinates used here represent eight blocks per
 * cell. By default a range of 12 cells is sampled, which can be overriden for
 * optimization purposes.
 */
float getEndHeightNoise(const EndNoise *en, int x, int z, int range)
{
    int hx = x / 2;
    int hz = z / 2;
    int oddx = x % 2;
    int oddz = z % 2;
    int i, j;

    int64_t h = 64 * (x*(int64_t)x + z*(int64_t)z);
    if (range == 0)
        range = 12;

    for (j = -range; j <= range; j++)
    {
        for (i = -range; i <= range; i++)
        {
            int64_t rx = hx + i;
            int64_t rz = hz + j;
            uint64_t rsq = rx*rx + rz*rz;
            uint16_t v = 0;
            if (rsq > 4096 && sampleSimplex2D(&en->perlin, rx, rz) < -0.9f)
            {
                //v = (llabs(rx) * 3439 + llabs(rz) * 147) % 13 + 9;
                v = (unsigned int)(
                        fabsf((float)rx) * 3439.0f + fabsf((float)rz) * 147.0f
                    ) % 13 + 9;
                rx = (oddx - i * 2);
                rz = (oddz - j * 2);
                rsq = rx*rx + rz*rz;
                int64_t noise = rsq * v*v;
                if (noise < h)
                    h = noise;
            }
        }
    }

    float ret = 100 - sqrtf((float) h);
    if (ret < -100) ret = -100;
    if (ret > 80) ret = 80;
    return ret;
}

void sampleNoiseColumnEnd(double column[],
    const SurfaceNoise *sn, const EndNoise *en, int x, int z,
    int colymin, int colymax)
{
    // clamped (32 + 46 - y) / 64.0
    static const double upper_drop[] = {
           1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0, // 0-7
           1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    1.0, 63./64, // 8-15
        62./64, 61./64, 60./64, 59./64, 58./64, 57./64, 56./64, 55./64, // 16-23
        54./64, 53./64, 52./64, 51./64, 50./64, 49./64, 48./64, 47./64, // 24-31
        46./64 // 32
    };
    // clamped (y - 1) / 7.0
    static const double lower_drop[] = {
          0.0,  0.0, 1./7, 2./7, 3./7, 4./7, 5./7, 6./7, // 0-7
          1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, // 8-15
          1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, // 16-23
          1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, // 24-31
          1.0, // 32
    };

    int y;
    if (en->mc > MC_1_13)
    {   // add outer end rings
        uint64_t rsq = (uint64_t) x * x + (uint64_t) z * z;
        if ((int)rsq < 0)
        {
            for (y = colymin; y <= colymax; y++)
                column[y - colymin] = nan("");
            return;
        }
    }

    // depth is between [-108, +72]
    // noise is between [-128, +128]
    // for a sold block we need the upper drop as:
    //  (72 + 128) * u - 3000 * (1-u) > 0 => upper_drop = u < 15/16
    // which occurs at y = 18 for the highest relevant noise cell
    // for the lower drop we need:
    //  (72 + 128) * l - 30 * (1-l) > 0 => lower_drop = l > 3/23
    // which occurs at y = 3 for the lowest relevant noise cell

    double depth = getEndHeightNoise(en, x, z, 0) - 8.0f;
    for (y = colymin; y <= colymax; y++)
    {
        if (lower_drop[y] == 0.0) {
            column[y - colymin] = -30;
            continue;
        }
        double noise = sampleSurfaceNoiseBetween(sn, x, y, z, -128, +128);
        double clamped = noise + depth;
        clamped = lerp(upper_drop[y], -3000, clamped);
        clamped = lerp(lower_drop[y], -30, clamped);
        column[y - colymin] = clamped;
    }
}

/* Given bordering noise columns and a fractional position between those,
 * determine the surface block height (i.e. where the interpolated noise > 0).
 * Note that the noise columns should be of size: ncolxz[ colymax-colymin+1 ]
 */
int getSurfaceHeight(
        const double ncol00[], const double ncol01[],
        const double ncol10[], const double ncol11[],
        int colymin, int colymax, int blockspercell, double dx, double dz)
{
    int y, celly;
    for (celly = colymax-1; celly >= colymin; celly--)
    {
        int idx = celly - colymin;
        double v000 = ncol00[idx];
        double v001 = ncol01[idx];
        double v100 = ncol10[idx];
        double v101 = ncol11[idx];
        double v010 = ncol00[idx+1];
        double v011 = ncol01[idx+1];
        double v110 = ncol10[idx+1];
        double v111 = ncol11[idx+1];

        for (y = blockspercell - 1; y >= 0; y--)
        {
            double dy = y / (double) blockspercell;
            double noise = lerp3(dy, dx, dz, // Note: not x, y, z
                v000, v010, v100, v110,
                v001, v011, v101, v111);
            if (noise > 0)
                return celly * blockspercell + y;
        }
    }
    return 0;
}

int getEndSurfaceHeight(int mc, uint64_t seed, int x, int z)
{
    EndNoise en;
    setEndSeed(&en, mc, seed);

    SurfaceNoise sn;
    initSurfaceNoise(&sn, DIM_END, seed);

    // end noise columns vary on a grid of cell size = eight
    int cellx = (x >> 3);
    int cellz = (z >> 3);
    double dx = (x & 7) / 8.0;
    double dz = (z & 7) / 8.0;

    // abusing enum for local compile time constants rather than enumeration
    enum { y0 = 0, y1 = 32, yn = y1-y0+1 };
    double ncol00[yn];
    double ncol01[yn];
    double ncol10[yn];
    double ncol11[yn];
    sampleNoiseColumnEnd(ncol00, &sn, &en, cellx, cellz, y0, y1);
    sampleNoiseColumnEnd(ncol01, &sn, &en, cellx, cellz+1, y0, y1);
    sampleNoiseColumnEnd(ncol10, &sn, &en, cellx+1, cellz, y0, y1);
    sampleNoiseColumnEnd(ncol11, &sn, &en, cellx+1, cellz+1, y0, y1);

    return getSurfaceHeight(ncol00, ncol01, ncol10, ncol11, y0, y1, 4, dx, dz);
}

int mapEndSurfaceHeight(float *y, const EndNoise *en, const SurfaceNoise *sn,
    int x, int z, int w, int h, int scale, int ymin)
{
    if (scale != 1 && scale != 2 && scale != 4 && scale != 8)
        return 1;

    int y0 = ymin >> 2;
    if (y0 <  2) y0 =  2;
    if (y0 > 17) y0 = 17;
    int y1 = 18;
    int yn = y1 - y0 + 1;
    double cellmid = scale > 1 ? scale / 16.0 : 0;
    int cellsiz = 8 / scale;
    int cx = floordiv(x, cellsiz);
    int cz = floordiv(z, cellsiz);
    int cw = floordiv(x + w - 1, cellsiz) - cx + 2;
    int i, j;

    double *buf = malloc(sizeof(double) * yn * cw * 2);
    double *ncol[2];
    ncol[0] = buf;
    ncol[1] = buf + yn * cw;

    for (i = 0; i < cw; i++)
        sampleNoiseColumnEnd(ncol[1]+i*yn, sn, en, cx+i, cz+0, y0, y1);

    for (j = 0; j < h; j++)
    {
        int cj = floordiv(z + j, cellsiz);
        int dj = z + j - cj * cellsiz;
        if (j == 0 || dj == 0)
        {
            double *tmp = ncol[0];
            ncol[0] = ncol[1];
            ncol[1] = tmp;
            for (i = 0; i < cw; i++)
                sampleNoiseColumnEnd(ncol[1]+i*yn, sn, en, cx+i, cj+1, y0, y1);
        }

        for (i = 0; i < w; i++)
        {
            int ci = floordiv(x + i, cellsiz);
            int di = x + i - ci * cellsiz;
            double dx = di / (double) cellsiz + cellmid;
            double dz = dj / (double) cellsiz + cellmid;
            double *ncol0 = ncol[0] + (ci - cx) * yn;
            double *ncol1 = ncol[1] + (ci - cx) * yn;
            y[j*w+i] = getSurfaceHeight(ncol0, ncol1, ncol0+yn, ncol1+yn,
                y0, y1, 4, dx, dz);
        }
    }

    free(buf);
    return 0;
}

int genEndScaled(const EndNoise *en, int *out, Range r, int mc, uint64_t sha)
{
    if (mc < MC_1_0)
        return 1;
    if (r.sy == 0)
        r.sy = 1;

    if (mc <= MC_1_8)
    {
        uint64_t i, siz = (uint64_t)r.sx*r.sy*r.sz;
        for (i = 0; i < siz; i++)
            out[i] = the_end;
        return 0;
    }

    int err, iy;

    if (r.scale == 1)
    {
        Range s = getVoronoiSrcRange(r);
        err = mapEnd(en, out, s.x, s.z, s.sx, s.sz);
        if (err) return err;

        if (mc <= MC_1_14)
        {   // up to 1.14 voronoi noise is planar
            Layer lvoronoi;
            memset(&lvoronoi, 0, sizeof(Layer));
            lvoronoi.startSalt = getLayerSalt(10);
            err = mapVoronoi114(&lvoronoi, out, r.x, r.z, r.sx, r.sz);
            if (err) return err;
        }
        else
        {   // in 1.15 voronoi noise varies vertically in the End
            int *src = out + (int64_t)r.sx*r.sy*r.sz;
            memmove(src, out, sizeof(int)*s.sx*s.sz);
            for (iy = 0; iy < r.sy; iy++)
            {
                mapVoronoiPlane(
                    sha, out+r.sx*r.sz*iy, src,
                    r.x,r.z,r.sx,r.sz, r.y+iy,
                    s.x,s.z,s.sx,s.sz);
            }
            return 0; // 3D expansion is done => return
        }
    }
    else if (r.scale == 4)
    {
        err = mapEnd(en, out, r.x, r.z, r.sx, r.sz);
        if (err) return err;
    }
    else if (r.scale == 16)
    {
        err = mapEndBiome(en, out, r.x, r.z, r.sx, r.sz);
        if (err) return err;
    }
    else
    {
        float d = r.scale / 8.0;
        int i, j;
        for (j = 0; j < r.sz; j++)
        {
            for (i = 0; i < r.sx; i++)
            {
                int64_t hx = (int64_t)((i+r.x) * d);
                int64_t hz = (int64_t)((j+r.z) * d);
                uint64_t rsq = hx*hx + hz*hz;
                if (rsq <= 16384L)
                {
                    out[j*r.sx+i] = the_end;
                    continue;
                }
                else if (mc > MC_1_13 && (int)(rsq) < 0)
                {
                    out[j*r.sx+i] = end_barrens;
                    continue;
                }
                float h = getEndHeightNoise(en, hx, hz, 4);
                if (h > 40)
                    out[j*r.sx+i] = end_highlands;
                else if (h >= 0)
                    out[j*r.sx+i] = end_midlands;
                else if (h >= -20)
                    out[j*r.sx+i] = end_barrens;
                else
                    out[j*r.sx+i] = small_end_islands;
            }
        }
    }

    // expanding 2D into 3D
    for (iy = 1; iy < r.sy; iy++)
    {
        int64_t i, siz = (int64_t)r.sx*r.sz;
        for (i = 0; i < siz; i++)
            out[iy*siz + i] = out[i];
    }

    return 0;
}


//==============================================================================
// Overworld and Nether Biome Generation 1.18
//==============================================================================

static int init_climate_seed(
    DoublePerlinNoise *dpn, PerlinNoise *oct,
    uint64_t xlo, uint64_t xhi, int large, int nptype, int nmax
    )
{
    Xoroshiro pxr;
    int n = 0;

    switch (nptype)
    {
    case NP_SHIFT: {
        static const double amp[] = {1, 1, 1, 0};
        // md5 "minecraft:offset"
        pxr.lo = xlo ^ 0x080518cf6af25384;
        pxr.hi = xhi ^ 0x3f3dfb40a54febd5;
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, -3, 4, nmax);
        } break;

    case NP_TEMPERATURE: {
        static const double amp[] = {1.5, 0, 1, 0, 0, 0};
        // md5 "minecraft:temperature" or "minecraft:temperature_large"
        pxr.lo = xlo ^ (large ? 0x944b0073edf549db : 0x5c7e6b29735f0d7f);
        pxr.hi = xhi ^ (large ? 0x4ff44347e9d22b96 : 0xf7d86f1bbc734988);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -12 : -10, 6, nmax);
        } break;

    case NP_HUMIDITY: {
        static const double amp[] = {1, 1, 0, 0, 0, 0};
        // md5 "minecraft:vegetation" or "minecraft:vegetation_large"
        pxr.lo = xlo ^ (large ? 0x71b8ab943dbd5301 : 0x81bb4d22e8dc168e);
        pxr.hi = xhi ^ (large ? 0xbb63ddcf39ff7a2b : 0xf1c8b4bea16303cd);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -10 : -8, 6, nmax);
        } break;

    case NP_CONTINENTALNESS: {
        static const double amp[] = {1, 1, 2, 2, 2, 1, 1, 1, 1};
        // md5 "minecraft:continentalness" or "minecraft:continentalness_large"
        pxr.lo = xlo ^ (large ? 0x9a3f51a113fce8dc : 0x83886c9d0ae3a662);
        pxr.hi = xhi ^ (large ? 0xee2dbd157e5dcdad : 0xafa638a61b42e8ad);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -11 : -9, 9, nmax);
        } break;

    case NP_EROSION: {
        static const double amp[] = {1, 1, 0, 1, 1};
        // md5 "minecraft:erosion" or "minecraft:erosion_large"
        pxr.lo = xlo ^ (large ? 0x8c984b1f8702a951 : 0xd02491e6058f6fd8);
        pxr.hi = xhi ^ (large ? 0xead7b1f92bae535f : 0x4792512c94c17a80);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -11 : -9, 5, nmax);
        } break;

    case NP_WEIRDNESS: {
        static const double amp[] = {1, 2, 1, 0, 0, 0};
        // md5 "minecraft:ridge"
        pxr.lo = xlo ^ 0xefc8ef4d36102b34;
        pxr.hi = xhi ^ 0x1beeeb324a0f24ea;
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, -7, 6, nmax);
        } break;

    default:
        printf("unsupported climate parameter %d\n", nptype);
        exit(1);
    }
    return n;
}

static void initLegacyXoroshiroOctaves(
    OctaveNoise *noise, Xoroshiro *xr, PerlinNoise *octaves, int count)
{
    int i;
    for (i = 0; i < count; i++)
        xPerlinInit(&octaves[i], xr);
    noise->octaves = octaves;
    noise->octcnt = count;
}

static void initModernTerrainNoise(SurfaceNoise *sn,
    uint64_t xlo, uint64_t xhi)
{
    // Xoroshiro random splitter salt for "minecraft:terrain".
    Xoroshiro xr = {
        xlo ^ 0x1ee555222ef96f14ULL,
        xhi ^ 0xe2bedfdbebe43d33ULL,
    };
    initLegacyXoroshiroOctaves(&sn->octmin, &xr, sn->oct + 0, 16);
    initLegacyXoroshiroOctaves(&sn->octmax, &xr, sn->oct + 16, 16);
    initLegacyXoroshiroOctaves(&sn->octmain, &xr, sn->oct + 32, 8);
    sn->xzScale = 0.25;
    sn->yScale = 0.125;
    sn->xzFactor = 80.0;
    sn->yFactor = 160.0;
}

static void initNamedTerrainNoise(
    BiomeNoise *bn, int index, uint64_t xlo, uint64_t xhi,
    uint64_t saltLo, uint64_t saltHi,
    const double *amplitudes, int firstOctave, int length)
{
    Xoroshiro xr = { xlo ^ saltLo, xhi ^ saltHi };
    xDoublePerlinInit(&bn->terrainNoise[index], &xr,
        bn->terrainNoiseOct[index], amplitudes, firstOctave, length, -1);
}

static void initNamedAquiferNoise(
    BiomeNoise *bn, int index, uint64_t xlo, uint64_t xhi,
    uint64_t saltLo, uint64_t saltHi, int firstOctave)
{
    static const double one[] = { 1.0 };
    Xoroshiro xr = { xlo ^ saltLo, xhi ^ saltHi };
    xDoublePerlinInit(&bn->aquiferNoise[index], &xr,
        bn->aquiferNoiseOct[index], one, firstOctave, 1, -1);
}

void setBiomeSeed(BiomeNoise *bn, uint64_t seed, int large)
{
    Xoroshiro pxr;
    xSetSeed(&pxr, seed);
    uint64_t xlo = xNextLong(&pxr);
    uint64_t xhi = xNextLong(&pxr);

    int n = 0, i = 0;
    for (; i < NP_MAX; i++)
        n += init_climate_seed(&bn->climate[i], bn->oct+n, xlo, xhi, large, i, -1);

    // Xoroshiro random splitter salt for "minecraft:jagged".
    Xoroshiro jagged = {
        xlo ^ 0xf902c0a7c9daa994ULL,
        xhi ^ 0x71ecd96a8da5e503ULL,
    };
    static const double jaggedAmplitude[16] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    xDoublePerlinInit(&bn->jagged, &jagged, bn->jaggedOct,
        jaggedAmplitude, -16, 16, -1);
    initModernTerrainNoise(&bn->terrain, xlo, xhi);

    static const double one[] = { 1.0 };
    static const double caveEntrance[] = { 0.4, 0.5, 1.0 };
    initNamedTerrainNoise(bn, TN_SPAGHETTI_3D_RARITY, xlo, xhi,
        0xee4780c98d93a439ULL, 0xbe45d334fdb76d2cULL, one, -11, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_3D_THICKNESS, xlo, xhi,
        0x897e1f20ad2d2b27ULL, 0xf4142f5a58d1d5abULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_3D_1, xlo, xhi,
        0xa5053f53ce3f5840ULL, 0x834fa38a3ded87b7ULL, one, -7, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_3D_2, xlo, xhi,
        0xb9a6367335a0ceefULL, 0xd61ccfd0fac10ac3ULL, one, -7, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_ROUGHNESS, xlo, xhi,
        0x7aed57fd327d6591ULL, 0xd495a3593e4d67bdULL, one, -5, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_ROUGHNESS_MODULATOR, xlo, xhi,
        0xe19387d2ceaf4eaaULL, 0xcacdcd34e1bc2003ULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_CAVE_ENTRANCE, xlo, xhi,
        0xf1008a17493d9a65ULL, 0xbd1ce09e8bc70ea0ULL,
        caveEntrance, -7, 3);
    initNamedTerrainNoise(bn, TN_NOODLE, xlo, xhi,
        0xd23ce12b0c37e44bULL, 0x66b1fdfbf6a474f3ULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_NOODLE_THICKNESS, xlo, xhi,
        0x2fa49bbe949d4212ULL, 0x5f82c95251d19891ULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_NOODLE_RIDGE_A, xlo, xhi,
        0x86172ac1315f6026ULL, 0x4a664470c7d7205fULL, one, -7, 1);
    initNamedTerrainNoise(bn, TN_NOODLE_RIDGE_B, xlo, xhi,
        0xeb232b4b89fdde91ULL, 0x031ee565ead84e5cULL, one, -7, 1);
    static const double pillar[] = { 1.0, 1.0 };
    static const double caveCheese[] = { 0.5, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0 };
    initNamedTerrainNoise(bn, TN_PILLAR, xlo, xhi,
        0xd4defd1400fa5347ULL, 0xccb6785451feaa1cULL, pillar, -7, 2);
    initNamedTerrainNoise(bn, TN_PILLAR_RARENESS, xlo, xhi,
        0x8ddd6be7cd4b24d0ULL, 0x0485e8665333197aULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_PILLAR_THICKNESS, xlo, xhi,
        0x1a28cb2542f8308dULL, 0xc4bba10b7fc7168cULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_2D, xlo, xhi,
        0x34748c98fa19c477ULL, 0x2a02051eb9e9b9cdULL, one, -7, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_2D_ELEVATION, xlo, xhi,
        0x29dd5fcf38d9edcdULL, 0x5bb1d7a839f6d4abULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_2D_MODULATOR, xlo, xhi,
        0x95e07097e4685522ULL, 0x0232deaf95eb5fedULL, one, -11, 1);
    initNamedTerrainNoise(bn, TN_SPAGHETTI_2D_THICKNESS, xlo, xhi,
        0x37e5cb432b85f4c8ULL, 0xe5e16b35b407aebcULL, one, -11, 1);
    initNamedTerrainNoise(bn, TN_CAVE_LAYER, xlo, xhi,
        0x4dfe67be2ef51a83ULL, 0x5a4ae8c4423e7206ULL, one, -8, 1);
    initNamedTerrainNoise(bn, TN_CAVE_CHEESE, xlo, xhi,
        0xb159093bc7baaa50ULL, 0x53abc45424417c20ULL,
        caveCheese, -8, 9);

    initNamedAquiferNoise(bn, AN_BARRIER, xlo, xhi,
        0xe170fda6c51d093fULL, 0x134f3e4bc3f40b0cULL, -3);
    initNamedAquiferNoise(bn, AN_FLOODEDNESS, xlo, xhi,
        0x1b5d4f85fdc2d89eULL, 0x22c7e3e611bf67c6ULL, -7);
    initNamedAquiferNoise(bn, AN_SPREAD, xlo, xhi,
        0x08f136f47cd9aaa2ULL, 0x3666182f5fa0ac6fULL, -5);
    initNamedAquiferNoise(bn, AN_LAVA, xlo, xhi,
        0x0e5121277ab51b72ULL, 0x2724d8c56aa005baULL, -1);
    Xoroshiro aquifer = {
        xlo ^ 0x7bb15cc403c6ace6ULL,
        xhi ^ 0x0bdd56bc9d232691ULL,
    };
    bn->aquiferSeedLo = xNextLong(&aquifer);
    bn->aquiferSeedHi = xNextLong(&aquifer);

    if ((size_t)n > sizeof(bn->oct) / sizeof(*bn->oct))
    {
        printf("setBiomeSeed(): BiomeNoise is malformed, buffer too small\n");
        exit(1);
    }
    bn->nptype = -1;
}

void setBetaBiomeSeed(BiomeNoiseBeta *bnb, uint64_t seed)
{
    uint64_t seedScratch;
    setSeed(&seedScratch, seed*9871);
    octaveInitBeta(bnb->climate, &seedScratch, bnb->oct,
        4, 0.025/1.5, 0.25, 0.55, 2.0);
    setSeed(&seedScratch, seed*39811);
    octaveInitBeta(bnb->climate+1, &seedScratch, bnb->oct+4,
        4, 0.05/1.5, 1./3, 0.55, 2.0);
    setSeed(&seedScratch, seed*0x84a59L);
    octaveInitBeta(bnb->climate+2, &seedScratch, bnb->oct+8,
        2, 0.25/1.5, 10./17, 0.55, 2.0);
    bnb->nptype = -1;
}

void addSplineVal(Spline *rsp, float loc, Spline *val, float der)
{
    rsp->loc[rsp->len] = loc;
    rsp->val[rsp->len] = val;
    rsp->der[rsp->len] = der;
    rsp->len++;
    //if (rsp->len > 12) {
    //    printf("addSplineVal(): too many spline points\n");
    //    exit(1);
    //}
}

Spline *createFixSpline(SplineStack *ss, float val)
{
    FixSpline *sp = &ss->fstack[ss->flen++];
    sp->len = 1;
    sp->val = val;
    return (Spline*)sp;
}

static float getOffsetValue(float weirdness, float continentalness)
{
    float f0 = 1.0F - (1.0F - continentalness) * 0.5F;
    float f1 = 0.5F * (1.0F - continentalness);
    float f2 = (weirdness + 1.17F) * 0.46082947F;
    float off = f2 * f0 - f1;
    if (weirdness < -0.7F)
        return off > -0.2222F ? off : -0.2222F;
    else
        return off > 0 ? off : 0;
}

static Spline *createSpline_38219(SplineStack *ss, float f, int bl)
{
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_RIDGES;

    float i = getOffsetValue(-1.0F, f);
    float k = getOffsetValue( 1.0F, f);
    float l = 1.0F - (1.0F - f) * 0.5F;
    float u = 0.5F * (1.0F - f);
    l = u / (0.46082947F * l) - 1.17F;

    if (-0.65F < l && l < 1.0F)
    {
        float p, q, r, s;
        u = getOffsetValue(-0.65F, f);
        p = getOffsetValue(-0.75F, f);
        q = (p - i) * 4.0F;
        r = getOffsetValue(l, f);
        s = (k - r) / (1.0F - l);

        addSplineVal(sp, -1.0F,     createFixSpline(ss, i), q);
        addSplineVal(sp, -0.75F,    createFixSpline(ss, p), 0);
        addSplineVal(sp, -0.65F,    createFixSpline(ss, u), 0);
        addSplineVal(sp, l-0.01F,   createFixSpline(ss, r), 0);
        addSplineVal(sp, l,         createFixSpline(ss, r), s);
        addSplineVal(sp, 1.0F,      createFixSpline(ss, k), s);
    }
    else
    {
        u = (k - i) * 0.5F;
        if (bl) {
            addSplineVal(sp, -1.0F, createFixSpline(ss, i > 0.2 ? i : 0.2), 0);
            addSplineVal(sp,  0.0F, createFixSpline(ss, lerp(0.5F, i, k)), u);
        } else {
            addSplineVal(sp, -1.0F, createFixSpline(ss, i), u);
        }
        addSplineVal(sp, 1.0F,      createFixSpline(ss, k), u);
    }
    return sp;
}

static Spline *createFlatOffsetSpline(
    SplineStack *ss, float f, float g, float h, float i, float j, float k)
{
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_RIDGES;

    float l = 0.5F * (g - f); if (l < k) l = k;
    float m = 5.0F * (h - g);

    addSplineVal(sp, -1.0F, createFixSpline(ss, f), l);
    addSplineVal(sp, -0.4F, createFixSpline(ss, g), l < m ? l : m);
    addSplineVal(sp,  0.0F, createFixSpline(ss, h), m);
    addSplineVal(sp,  0.4F, createFixSpline(ss, i), 2.0F*(i-h));
    addSplineVal(sp,  1.0F, createFixSpline(ss, j), 0.7F*(j-i));

    return sp;
}

static Spline *createLandSpline(
    SplineStack *ss, float f, float g, float h, float i, float j, float k, int bl)
{
    Spline *sp1 = createSpline_38219(ss, lerp(i, 0.6F, 1.5F), bl);
    Spline *sp2 = createSpline_38219(ss, lerp(i, 0.6F, 1.0F), bl);
    Spline *sp3 = createSpline_38219(ss, i, bl);
    const float ih = 0.5F * i;
    Spline *sp4 = createFlatOffsetSpline(ss, f-0.15F, ih, ih, ih, i*0.6F, 0.5F);
    Spline *sp5 = createFlatOffsetSpline(ss, f, j*i, g*i, ih, i*0.6F, 0.5F);
    Spline *sp6 = createFlatOffsetSpline(ss, f, j, j, g, h, 0.5F);
    Spline *sp7 = createFlatOffsetSpline(ss, f, j, j, g, h, 0.5F);

    Spline *sp8 = &ss->stack[ss->len++];
    sp8->typ = SP_RIDGES;
    addSplineVal(sp8, -1.0F, createFixSpline(ss, f), 0.0F);
    addSplineVal(sp8, -0.4F, sp6, 0.0F);
    addSplineVal(sp8,  0.0F, createFixSpline(ss, h + 0.07F), 0.0F);

    Spline *sp9 = createFlatOffsetSpline(ss, -0.02F, k, k, g, h, 0.0F);
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_EROSION;
    addSplineVal(sp, -0.85F, sp1, 0.0F);
    addSplineVal(sp, -0.7F,  sp2, 0.0F);
    addSplineVal(sp, -0.4F,  sp3, 0.0F);
    addSplineVal(sp, -0.35F, sp4, 0.0F);
    addSplineVal(sp, -0.1F,  sp5, 0.0F);
    addSplineVal(sp,  0.2F,  sp6, 0.0F);
    if (bl) {
        addSplineVal(sp, 0.4F,  sp7, 0.0F);
        addSplineVal(sp, 0.45F, sp8, 0.0F);
        addSplineVal(sp, 0.55F, sp8, 0.0F);
        addSplineVal(sp, 0.58F, sp7, 0.0F);
    }
    addSplineVal(sp, 0.7F, sp9, 0.0F);
    return sp;
}

static Spline *createFactorRidgeSpline(SplineStack *ss, float value)
{
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_WEIRDNESS;
    addSplineVal(sp, -0.2F, createFixSpline(ss, 6.3F), 0.0F);
    addSplineVal(sp,  0.2F, createFixSpline(ss, value), 0.0F);
    return sp;
}

static Spline *createFactorErosionSpline(SplineStack *ss, float value, int mountain)
{
    Spline *ridge = createFactorRidgeSpline(ss, value);
    Spline *low = &ss->stack[ss->len++];
    low->typ = SP_WEIRDNESS;
    addSplineVal(low, -0.05F, createFixSpline(ss, 6.3F), 0.0F);
    addSplineVal(low,  0.05F, createFixSpline(ss, 2.67F), 0.0F);
    Spline *high = &ss->stack[ss->len++];
    high->typ = SP_WEIRDNESS;
    addSplineVal(high, -0.05F, createFixSpline(ss, 2.67F), 0.0F);
    addSplineVal(high,  0.05F, createFixSpline(ss, 6.3F), 0.0F);

    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_EROSION;
    addSplineVal(sp, -0.6F, ridge, 0.0F);
    addSplineVal(sp, -0.5F, low, 0.0F);
    addSplineVal(sp, -0.35F, ridge, 0.0F);
    addSplineVal(sp, -0.25F, ridge, 0.0F);
    addSplineVal(sp, -0.1F, high, 0.0F);
    addSplineVal(sp,  0.03F, ridge, 0.0F);

    if (mountain)
    {
        Spline *ridgeTransition = &ss->stack[ss->len++];
        ridgeTransition->typ = SP_WEIRDNESS;
        addSplineVal(ridgeTransition, 0.0F, createFixSpline(ss, value), 0.0F);
        addSplineVal(ridgeTransition, 0.1F, createFixSpline(ss, 0.625F), 0.0F);
        Spline *folded = &ss->stack[ss->len++];
        folded->typ = SP_RIDGES;
        addSplineVal(folded, -0.9F, createFixSpline(ss, value), 0.0F);
        addSplineVal(folded, -0.69F, ridgeTransition, 0.0F);
        addSplineVal(sp, 0.35F, createFixSpline(ss, value), 0.0F);
        addSplineVal(sp, 0.45F, folded, 0.0F);
        addSplineVal(sp, 0.55F, folded, 0.0F);
        addSplineVal(sp, 0.62F, createFixSpline(ss, value), 0.0F);
    }
    else
    {
        Spline *lowerFolded = &ss->stack[ss->len++];
        lowerFolded->typ = SP_RIDGES;
        addSplineVal(lowerFolded, -0.7F, ridge, 0.0F);
        addSplineVal(lowerFolded, -0.15F, createFixSpline(ss, 1.37F), 0.0F);
        Spline *upperFolded = &ss->stack[ss->len++];
        upperFolded->typ = SP_RIDGES;
        addSplineVal(upperFolded, 0.45F, ridge, 0.0F);
        addSplineVal(upperFolded, 0.7F, createFixSpline(ss, 1.56F), 0.0F);
        addSplineVal(sp, 0.05F, upperFolded, 0.0F);
        addSplineVal(sp, 0.4F, upperFolded, 0.0F);
        addSplineVal(sp, 0.45F, lowerFolded, 0.0F);
        addSplineVal(sp, 0.55F, lowerFolded, 0.0F);
        addSplineVal(sp, 0.58F, createFixSpline(ss, value), 0.0F);
    }
    return sp;
}

static Spline *createFactorSpline(SplineStack *ss)
{
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_CONTINENTALNESS;
    addSplineVal(sp, -0.19F, createFixSpline(ss, 3.95F), 0.0F);
    addSplineVal(sp, -0.15F, createFactorErosionSpline(ss, 6.25F, 1), 0.0F);
    addSplineVal(sp, -0.10F, createFactorErosionSpline(ss, 5.47F, 1), 0.0F);
    addSplineVal(sp,  0.03F, createFactorErosionSpline(ss, 5.08F, 1), 0.0F);
    addSplineVal(sp,  0.06F, createFactorErosionSpline(ss, 4.69F, 0), 0.0F);
    return sp;
}

static inline float peaksAndValleys(float weirdness)
{
    return -(fabsf(fabsf(weirdness) - 0.6666667F) - 0.33333334F) * 3.0F;
}

static Spline *createJaggedRidgeValueSpline(
    SplineStack *ss, float ridgeValue, float middleValue)
{
    const float low = peaksAndValleys(0.4F);
    const float middle = (low + peaksAndValleys(0.56666666F)) * 0.5F;
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_RIDGES;
    addSplineVal(sp, low, createFixSpline(ss, 0.0F), 0.0F);
    if (middleValue > 0.0F)
    {
        Spline *raw = &ss->stack[ss->len++];
        raw->typ = SP_WEIRDNESS;
        addSplineVal(raw, -0.01F, createFixSpline(ss, 0.63F * middleValue), 0.0F);
        addSplineVal(raw,  0.01F, createFixSpline(ss, 0.30F * middleValue), 0.0F);
        addSplineVal(sp, middle, raw, 0.0F);
    }
    else
    {
        addSplineVal(sp, middle, createFixSpline(ss, 0.0F), 0.0F);
    }
    if (ridgeValue > 0.0F)
    {
        Spline *raw = &ss->stack[ss->len++];
        raw->typ = SP_WEIRDNESS;
        addSplineVal(raw, -0.01F, createFixSpline(ss, 0.63F * ridgeValue), 0.0F);
        addSplineVal(raw,  0.01F, createFixSpline(ss, 0.30F * ridgeValue), 0.0F);
        addSplineVal(sp, 1.0F, raw, 0.0F);
    }
    else
    {
        addSplineVal(sp, 1.0F, createFixSpline(ss, 0.0F), 0.0F);
    }
    return sp;
}

static Spline *createJaggedErosionSpline(
    SplineStack *ss, float lowRidge, float highRidge,
    float lowMiddle, float highMiddle)
{
    Spline *low = createJaggedRidgeValueSpline(ss, lowRidge, lowMiddle);
    Spline *high = createJaggedRidgeValueSpline(ss, highRidge, highMiddle);
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_EROSION;
    addSplineVal(sp, -1.0F, low, 0.0F);
    addSplineVal(sp, -0.78F, high, 0.0F);
    addSplineVal(sp, -0.5775F, high, 0.0F);
    addSplineVal(sp, -0.375F, createFixSpline(ss, 0.0F), 0.0F);
    return sp;
}

static Spline *createJaggednessSpline(SplineStack *ss)
{
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_CONTINENTALNESS;
    addSplineVal(sp, -0.11F, createFixSpline(ss, 0.0F), 0.0F);
    addSplineVal(sp, 0.03F,
        createJaggedErosionSpline(ss, 1.0F, 0.5F, 0.0F, 0.0F), 0.0F);
    addSplineVal(sp, 0.65F,
        createJaggedErosionSpline(ss, 1.0F, 1.0F, 1.0F, 0.0F), 0.0F);
    return sp;
}

float getSpline(const Spline *sp, const float *vals)
{
    if (!sp || sp->len <= 0 || sp->len >= 12)
    {
        printf("getSpline(): bad parameters\n");
        exit(1);
    }

    if (sp->len == 1)
        return ((FixSpline*)sp)->val;

    float f = vals[sp->typ];
    int i;

    for (i = 0; i < sp->len; i++)
        if (sp->loc[i] >= f)
            break;
    if (i == 0 || i == sp->len)
    {
        if (i) i--;
        float v = getSpline(sp->val[i], vals);
        return v + sp->der[i] * (f - sp->loc[i]);
    }
    const Spline *sp1 = sp->val[i-1];
    const Spline *sp2 = sp->val[i];
    float g = sp->loc[i-1];
    float h = sp->loc[i];
    float k = (f - g) / (h - g);
    float l = sp->der[i-1];
    float m = sp->der[i];
    float n = getSpline(sp1, vals);
    float o = getSpline(sp2, vals);
    float p = l * (h - g) - (o - n);
    float q = -m * (h - g) + (o - n);
    float r = lerp(k, n, o) + k * (1.0F - k) * lerp(k, p, q);
    return r;
}

void initBiomeNoise(BiomeNoise *bn, int mc)
{
    SplineStack *ss = &bn->ss;
    memset(ss, 0, sizeof(*ss));
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_CONTINENTALNESS;

    Spline *sp1 = createLandSpline(ss, -0.15F, 0.00F, 0.0F, 0.1F, 0.00F, -0.03F, 0);
    Spline *sp2 = createLandSpline(ss, -0.10F, 0.03F, 0.1F, 0.1F, 0.01F, -0.03F, 0);
    Spline *sp3 = createLandSpline(ss, -0.10F, 0.03F, 0.1F, 0.7F, 0.01F, -0.03F, 1);
    Spline *sp4 = createLandSpline(ss, -0.05F, 0.03F, 0.1F, 1.0F, 0.01F,  0.01F, 1);

    addSplineVal(sp, -1.10F, createFixSpline(ss,  0.044F), 0.0F);
    addSplineVal(sp, -1.02F, createFixSpline(ss, -0.2222F), 0.0F);
    addSplineVal(sp, -0.51F, createFixSpline(ss, -0.2222F), 0.0F);
    addSplineVal(sp, -0.44F, createFixSpline(ss, -0.12F), 0.0F);
    addSplineVal(sp, -0.18F, createFixSpline(ss, -0.12F), 0.0F);
    addSplineVal(sp, -0.16F, sp1, 0.0F);
    addSplineVal(sp, -0.15F, sp1, 0.0F);
    addSplineVal(sp, -0.10F, sp2, 0.0F);
    addSplineVal(sp,  0.25F, sp3, 0.0F);
    addSplineVal(sp,  1.00F, sp4, 0.0F);

    bn->sp = sp;
    memset(&bn->factorSs, 0, sizeof(bn->factorSs));
    bn->factorSp = createFactorSpline(&bn->factorSs);
    memset(&bn->jaggedSs, 0, sizeof(bn->jaggedSs));
    bn->jaggedSp = createJaggednessSpline(&bn->jaggedSs);
    bn->mc = mc;
}


/// Biome sampler for MC 1.18
int sampleBiomeNoise(const BiomeNoise *bn, int64_t *np, int x, int y, int z,
    uint64_t *dat, uint32_t sample_flags)
{
    if (bn->nptype >= 0)
    {   // initialized for a specific climate parameter
        if (np)
            memset(np, 0, NP_MAX*sizeof(*np));
        int64_t id = (int64_t) (10000.0 * sampleClimatePara(bn, np, x, z));
        return (int) id;
    }

    float t = 0, h = 0, c = 0, e = 0, d = 0, w = 0;
    double px = x, pz = z;
    if (!(sample_flags & SAMPLE_NO_SHIFT))
    {
        px += sampleDoublePerlin(&bn->climate[NP_SHIFT], x, 0, z) * 4.0;
        pz += sampleDoublePerlin(&bn->climate[NP_SHIFT], z, x, 0) * 4.0;
    }

    c = sampleDoublePerlin(&bn->climate[NP_CONTINENTALNESS], px, 0, pz);
    e = sampleDoublePerlin(&bn->climate[NP_EROSION], px, 0, pz);
    w = sampleDoublePerlin(&bn->climate[NP_WEIRDNESS], px, 0, pz);

    if (!(sample_flags & SAMPLE_NO_DEPTH))
    {
        float np_param[] = {
            c, e, peaksAndValleys(w), w,
        };
        double off = getSpline(bn->sp, np_param) + 0.015F;

        //double py = y + sampleDoublePerlin(&bn->shift, y, z, x) * 4.0;
        d = 1.0 - (y * 4) / 128.0 - 83.0/160.0 + off;
    }

    t = sampleDoublePerlin(&bn->climate[NP_TEMPERATURE], px, 0, pz);
    h = sampleDoublePerlin(&bn->climate[NP_HUMIDITY], px, 0, pz);

    int64_t l_np[6];
    int64_t *p_np = np ? np : l_np;
    p_np[0] = (int64_t)(10000.0F*t);
    p_np[1] = (int64_t)(10000.0F*h);
    p_np[2] = (int64_t)(10000.0F*c);
    p_np[3] = (int64_t)(10000.0F*e);
    p_np[4] = (int64_t)(10000.0F*d);
    p_np[5] = (int64_t)(10000.0F*w);

    int id = none;
    if (!(sample_flags & SAMPLE_NO_BIOME))
        id = climateToBiome(bn->mc, (const uint64_t*)p_np, dat);
    return id;
}

static void sampleTerrainParameters(
    const BiomeNoise *bn, double x, double z, float values[4])
{
    double px = x;
    double pz = z;
    px += sampleDoublePerlin(&bn->climate[NP_SHIFT], x, 0, z) * 4.0;
    pz += sampleDoublePerlin(&bn->climate[NP_SHIFT], z, x, 0) * 4.0;
    const float continentalness = sampleDoublePerlin(
        &bn->climate[NP_CONTINENTALNESS], px, 0, pz);
    const float erosion = sampleDoublePerlin(
        &bn->climate[NP_EROSION], px, 0, pz);
    const float weirdness = sampleDoublePerlin(
        &bn->climate[NP_WEIRDNESS], px, 0, pz);
    values[SP_CONTINENTALNESS] = continentalness;
    values[SP_EROSION] = erosion;
    values[SP_RIDGES] = peaksAndValleys(weirdness);
    values[SP_WEIRDNESS] = weirdness;
}

static double maintainPrecisionModern(double value)
{
    return value - floor(value / 33554432.0 + 0.5) * 33554432.0;
}

static double sampleModernTerrainNoise(
    const SurfaceNoise *sn, int x, int y, int z)
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
    double frequency = 1.0;
    int i;

    for (i = 0; i < 8; i++)
    {
        const PerlinNoise *noise = &sn->octmain.octaves[i];
        interpolation += samplePerlin(noise,
            maintainPrecisionModern(mainX * frequency),
            maintainPrecisionModern(mainY * frequency),
            maintainPrecisionModern(mainZ * frequency),
            mainSmearY * frequency,
            mainY * frequency) / frequency;
        frequency *= 0.5;
    }

    const double blend = (interpolation / 10.0 + 1.0) * 0.5;
    double lower = 0.0;
    double upper = 0.0;
    frequency = 1.0;
    for (i = 0; i < 16; i++)
    {
        const double sx = maintainPrecisionModern(dx * frequency);
        const double sy = maintainPrecisionModern(dy * frequency);
        const double sz = maintainPrecisionModern(dz * frequency);
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

static double sampleSlopedCheese(
    const BiomeNoise *bn, int blockX, int blockY, int blockZ,
    double offset, double factor, double jagged)
{
    const double depth = 1.0 - blockY / 128.0 + offset + jagged;
    const double scaledDepth = depth * factor;
    const double initialDensity = scaledDepth < 0.0
        ? scaledDepth
        : 4.0 * scaledDepth;
    return initialDensity + sampleModernTerrainNoise(
        &bn->terrain, blockX, blockY, blockZ);
}

static double clampValue(double value, double min, double max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static double mapNoiseRange(double value, double min, double max)
{
    return (min + max) * 0.5 + (max - min) * 0.5 * value;
}

static double sampleTerrainNoise(const BiomeNoise *bn, int index,
    double x, double y, double z)
{
    return sampleDoublePerlin(&bn->terrainNoise[index], x, y, z);
}

static double scaleTunnelRarity(double value)
{
    if (value < -0.5) return 0.75;
    if (value < 0.0) return 1.0;
    if (value < 0.5) return 1.5;
    return 2.0;
}

static double scaleCaveRarity(double value)
{
    if (value < -0.75) return 0.5;
    if (value < -0.5) return 0.75;
    if (value < 0.5) return 1.0;
    if (value < 0.75) return 2.0;
    return 3.0;
}

static double sampleCaveEntrance(
    const BiomeNoise *bn, int x, int y, int z)
{
    const double rarityNoise = sampleTerrainNoise(
        bn, TN_SPAGHETTI_3D_RARITY, x * 2.0, y, z * 2.0);
    const double rarity = scaleTunnelRarity(rarityNoise);
    const double spaghetti1 = rarity * fabs(sampleTerrainNoise(
        bn, TN_SPAGHETTI_3D_1, x / rarity, y / rarity, z / rarity));
    const double spaghetti2 = rarity * fabs(sampleTerrainNoise(
        bn, TN_SPAGHETTI_3D_2, x / rarity, y / rarity, z / rarity));
    const double thickness = mapNoiseRange(sampleTerrainNoise(
        bn, TN_SPAGHETTI_3D_THICKNESS, x, y, z), -0.065, -0.088);
    const double spaghetti = clampValue(
        (spaghetti1 > spaghetti2 ? spaghetti1 : spaghetti2) + thickness,
        -1.0, 1.0);

    const double roughnessNoise = sampleTerrainNoise(
        bn, TN_SPAGHETTI_ROUGHNESS, x, y, z);
    const double roughnessModulator = mapNoiseRange(sampleTerrainNoise(
        bn, TN_SPAGHETTI_ROUGHNESS_MODULATOR, x, y, z), 0.0, -0.1);
    const double roughness = roughnessModulator * (fabs(roughnessNoise) - 0.4);

    double gradient;
    if (y <= -10) gradient = 0.3;
    else if (y >= 30) gradient = 0.0;
    else gradient = 0.3 * (30 - y) / 40.0;
    const double entrance = sampleTerrainNoise(
        bn, TN_CAVE_ENTRANCE, x * 0.75, y * 0.5, z * 0.75)
        + 0.37 + gradient;
    const double tunnel = roughness + spaghetti;
    return entrance < tunnel ? entrance : tunnel;
}

static double sampleSpaghettiRoughness(
    const BiomeNoise *bn, int x, int y, int z)
{
    const double roughnessNoise = sampleTerrainNoise(
        bn, TN_SPAGHETTI_ROUGHNESS, x, y, z);
    const double roughnessModulator = mapNoiseRange(sampleTerrainNoise(
        bn, TN_SPAGHETTI_ROUGHNESS_MODULATOR, x, y, z), 0.0, -0.1);
    return roughnessModulator * (fabs(roughnessNoise) - 0.4);
}

static double sampleSpaghetti2d(
    const BiomeNoise *bn, int x, int y, int z)
{
    const double modulator = sampleTerrainNoise(
        bn, TN_SPAGHETTI_2D_MODULATOR, x * 2.0, y, z * 2.0);
    const double rarity = scaleCaveRarity(modulator);
    const double spaghetti = rarity * fabs(sampleTerrainNoise(
        bn, TN_SPAGHETTI_2D, x / rarity, y / rarity, z / rarity));
    const double elevation = mapNoiseRange(sampleTerrainNoise(
        bn, TN_SPAGHETTI_2D_ELEVATION, x, 0.0, z), -8.0, 8.0);
    double gradient;
    if (y <= -64) gradient = 8.0;
    else if (y >= 320) gradient = -40.0;
    else gradient = 8.0 - (y + 64) / 8.0;
    const double thickness = mapNoiseRange(sampleTerrainNoise(
        bn, TN_SPAGHETTI_2D_THICKNESS, x * 2.0, y, z * 2.0),
        -0.6, -1.3);
    const double vertical = fabs(elevation + gradient) + thickness;
    const double verticalCube = vertical * vertical * vertical;
    const double horizontal = spaghetti + 0.083 * thickness;
    return clampValue(horizontal > verticalCube ? horizontal : verticalCube,
        -1.0, 1.0);
}

static double samplePillar(
    const BiomeNoise *bn, int x, int y, int z)
{
    const double pillar = sampleTerrainNoise(
        bn, TN_PILLAR, x * 25.0, y * 0.3, z * 25.0);
    const double rareness = mapNoiseRange(sampleTerrainNoise(
        bn, TN_PILLAR_RARENESS, x, y, z), 0.0, -2.0);
    const double thickness = mapNoiseRange(sampleTerrainNoise(
        bn, TN_PILLAR_THICKNESS, x, y, z), 0.0, 1.1);
    return (2.0 * pillar + rareness) * thickness * thickness * thickness;
}

static double sampleCaves(
    const BiomeNoise *bn, int x, int y, int z, double slopedCheese)
{
    const double caveLayerNoise = sampleTerrainNoise(
        bn, TN_CAVE_LAYER, x, y * 8.0, z);
    const double caveLayer = 4.0 * caveLayerNoise * caveLayerNoise;
    const double cheeseNoise = sampleTerrainNoise(
        bn, TN_CAVE_CHEESE, x, y * (2.0/3.0), z);
    const double cheese = clampValue(0.27 + cheeseNoise, -1.0, 1.0)
        + clampValue(1.5 - 0.64 * slopedCheese, 0.0, 0.5);
    double caves = caveLayer + cheese;
    const double entrance = sampleCaveEntrance(bn, x, y, z);
    if (entrance < caves) caves = entrance;
    const double spaghetti = sampleSpaghetti2d(bn, x, y, z)
        + sampleSpaghettiRoughness(bn, x, y, z);
    if (spaghetti < caves) caves = spaghetti;

    const double pillar = samplePillar(bn, x, y, z);
    const double pillarChoice = pillar >= 0.03 ? pillar : -1000000.0;
    return pillarChoice > caves ? pillarChoice : caves;
}

static double sampleFinalTerrainDensity(
    const BiomeNoise *bn, int blockX, int blockY, int blockZ,
    double offset, double factor, double jagged)
{
    const double slopedCheese = sampleSlopedCheese(
        bn, blockX, blockY, blockZ, offset, factor, jagged);
    double density = slopedCheese;
    if (slopedCheese < 1.5625)
    {
        const double entrance = 5.0 * sampleCaveEntrance(
            bn, blockX, blockY, blockZ);
        if (entrance < density)
            density = entrance;
    }
    else
    {
        density = sampleCaves(
            bn, blockX, blockY, blockZ, slopedCheese);
    }
    return density;
}

enum {
    MODERN_MIN_CELL_Y = -8,
    MODERN_MAX_CELL_Y = 40,
    MODERN_COLUMN_SIZE = MODERN_MAX_CELL_Y - MODERN_MIN_CELL_Y + 1,
    MODERN_COLUMN_CACHE_LIMIT = 70000,
};

typedef struct {
    double density[MODERN_COLUMN_SIZE];
} ModernTerrainColumn;

static double applyModernSurfaceSlides(double density, int blockY)
{
    double alpha = (256.0 - blockY) / 16.0;
    density = clampedLerp(alpha, -0.078125, density);
    alpha = (blockY + 64.0) / 24.0;
    return clampedLerp(alpha, 0.1171875, density);
}

static void sampleModernTerrainColumn(
    ModernTerrainColumn *column, const BiomeNoise *bn, int cellX, int cellZ)
{
    const int blockX = cellX * 4;
    const int blockZ = cellZ * 4;
    float values[4];
    sampleTerrainParameters(bn, cellX, cellZ, values);
    const double offset = -0.50375F + getSpline(bn->sp, values);
    const double factor = getSpline(bn->factorSp, values);
    const double jaggedness = getSpline(bn->jaggedSp, values);
    double jagged = sampleDoublePerlin(&bn->jagged,
        blockX * 1500.0, 0.0, blockZ * 1500.0);
    if (jagged < 0.0)
        jagged *= 0.5;
    jagged *= jaggedness;

    int cellY;
    for (cellY = MODERN_MIN_CELL_Y; cellY <= MODERN_MAX_CELL_Y; cellY++)
    {
        const int blockY = cellY * 8;
        const double density = sampleFinalTerrainDensity(
            bn, blockX, blockY, blockZ, offset, factor, jagged);
        column->density[cellY - MODERN_MIN_CELL_Y] =
            applyModernSurfaceSlides(density, blockY);
    }
}

static void sampleNoodleCorner(
    const BiomeNoise *bn, int blockX, int blockY, int blockZ,
    double *selector, double *thickness, double *ridgeA, double *ridgeB)
{
    if (blockY < -60 || blockY >= 321)
    {
        *selector = -1.0;
        *thickness = 0.0;
        *ridgeA = 0.0;
        *ridgeB = 0.0;
        return;
    }
    *selector = sampleTerrainNoise(bn, TN_NOODLE,
        blockX, blockY, blockZ);
    *thickness = -0.075 - 0.025 * sampleTerrainNoise(
        bn, TN_NOODLE_THICKNESS, blockX, blockY, blockZ);
    *ridgeA = sampleTerrainNoise(bn, TN_NOODLE_RIDGE_A,
        blockX * (8.0/3.0), blockY * (8.0/3.0), blockZ * (8.0/3.0));
    *ridgeB = sampleTerrainNoise(bn, TN_NOODLE_RIDGE_B,
        blockX * (8.0/3.0), blockY * (8.0/3.0), blockZ * (8.0/3.0));
}

static double sampleInterpolatedNoodle(
    const BiomeNoise *bn, int cellX, int cellY, int cellZ,
    double dx, double dy, double dz)
{
    double selector[8], thickness[8], ridgeA[8], ridgeB[8];
    int ix, iy, iz;
    for (iz = 0; iz < 2; iz++)
    {
        for (ix = 0; ix < 2; ix++)
        {
            for (iy = 0; iy < 2; iy++)
            {
                const int index = iz*4 + ix*2 + iy;
                sampleNoodleCorner(bn,
                    (cellX + ix) * 4,
                    (cellY + iy) * 8,
                    (cellZ + iz) * 4,
                    selector + index, thickness + index,
                    ridgeA + index, ridgeB + index);
            }
        }
    }

    const double selectorValue = lerp3(dy, dx, dz,
        selector[0], selector[1], selector[2], selector[3],
        selector[4], selector[5], selector[6], selector[7]);
    if (selectorValue < 0.0)
        return 64.0;
    const double thicknessValue = lerp3(dy, dx, dz,
        thickness[0], thickness[1], thickness[2], thickness[3],
        thickness[4], thickness[5], thickness[6], thickness[7]);
    const double ridgeAValue = fabs(lerp3(dy, dx, dz,
        ridgeA[0], ridgeA[1], ridgeA[2], ridgeA[3],
        ridgeA[4], ridgeA[5], ridgeA[6], ridgeA[7]));
    const double ridgeBValue = fabs(lerp3(dy, dx, dz,
        ridgeB[0], ridgeB[1], ridgeB[2], ridgeB[3],
        ridgeB[4], ridgeB[5], ridgeB[6], ridgeB[7]));
    return thicknessValue + 1.5 *
        (ridgeAValue > ridgeBValue ? ridgeAValue : ridgeBValue);
}

static double sampleModernBaseDensity(
    const ModernTerrainColumn *column00,
    const ModernTerrainColumn *column01,
    const ModernTerrainColumn *column10,
    const ModernTerrainColumn *column11,
    int cellY,
    double dx, double dy, double dz)
{
    const int index = cellY - MODERN_MIN_CELL_Y;
    double density = lerp3(dy, dx, dz,
        column00->density[index], column00->density[index+1],
        column10->density[index], column10->density[index+1],
        column01->density[index], column01->density[index+1],
        column11->density[index], column11->density[index+1]);
    density = clampValue(density * 0.64, -1.0, 1.0);
    return density/2.0 - density*density*density/24.0;
}

static double sampleModernFinalDensity(
    const BiomeNoise *bn,
    const ModernTerrainColumn *column00,
    const ModernTerrainColumn *column01,
    const ModernTerrainColumn *column10,
    const ModernTerrainColumn *column11,
    int cellX, int cellY, int cellZ,
    double dx, double dy, double dz)
{
    const double density = sampleModernBaseDensity(
        column00, column01, column10, column11,
        cellY, dx, dy, dz);
    const double noodle = sampleInterpolatedNoodle(
        bn, cellX, cellY, cellZ, dx, dy, dz);
    return density < noodle ? density : noodle;
}

enum {
    AQUIFER_AIR,
    AQUIFER_WATER,
    AQUIFER_LAVA,
    AQUIFER_NO_FLUID_Y = -32512,
    AQUIFER_CACHE_LIMIT = 1000000,
};

typedef struct {
    int y;
    int state;
} AquiferFluidLevel;

typedef struct {
    int blockX;
    int blockY;
    int blockZ;
    AquiferFluidLevel fluid;
    unsigned char positionReady;
    unsigned char fluidReady;
} AquiferCenter;

typedef struct {
    const BiomeNoise *bn;
    int minCellX;
    int minCellY;
    int minCellZ;
    int sizeX;
    int sizeY;
    int sizeZ;
    AquiferCenter *centers;
} AquiferCache;

static int aquiferFluidStateAt(const AquiferFluidLevel *level, int blockY)
{
    return blockY < level->y ? level->state : AQUIFER_AIR;
}

static AquiferFluidLevel aquiferGlobalFluid(int blockY)
{
    AquiferFluidLevel level;
    if (blockY < -54)
    {
        level.y = -54;
        level.state = AQUIFER_LAVA;
    }
    else
    {
        level.y = 63;
        level.state = AQUIFER_WATER;
    }
    return level;
}

static double sampleAquiferNoise(
    const BiomeNoise *bn, int type, int blockX, int blockY, int blockZ)
{
    double x = blockX;
    double y = blockY;
    double z = blockZ;
    if (type == AN_BARRIER)
    {
        x *= 0.5;
        y *= 0.5;
        z *= 0.5;
    }
    else if (type == AN_FLOODEDNESS)
    {
        y *= 0.67;
    }
    else if (type == AN_SPREAD)
    {
        y *= 0.7142857142857143;
    }
    return sampleDoublePerlin(&bn->aquiferNoise[type], x, y, z);
}

static int estimateModernSurfaceHeight(
    const BiomeNoise *bn, int blockX, int blockZ)
{
    const int cellX = floordiv(blockX, 4);
    const int cellZ = floordiv(blockZ, 4);
    float values[4];
    sampleTerrainParameters(bn, cellX, cellZ, values);
    const double offset = -0.50375F + getSpline(bn->sp, values);
    const double factor = getSpline(bn->factorSp, values);
    int blockY;
    for (blockY = 320; blockY >= -64; blockY -= 8)
    {
        const double depth = 1.0 - blockY/128.0 + offset;
        const double scaledDepth = depth * factor;
        double density = scaledDepth < 0.0 ? scaledDepth : 4.0*scaledDepth;
        density = clampValue(density - 0.703125, -64.0, 64.0);
        density = applyModernSurfaceSlides(density, blockY);
        if (density > 0.390625)
            return blockY;
    }
    return INT32_MAX;
}

static int aquiferDeepDark(
    const BiomeNoise *bn, int blockX, int blockY, int blockZ)
{
    float values[4];
    sampleTerrainParameters(bn, blockX/4.0, blockZ/4.0, values);
    const double erosion = values[SP_EROSION];
    const double offset = -0.50375F + getSpline(bn->sp, values);
    const double depth = 1.0 - blockY/128.0 + offset;
    return erosion < -0.22499999403953552
        && depth > 0.8999999761581421;
}

static int aquiferNoiseBasedFluidY(
    const BiomeNoise *bn,
    int blockX, int blockY, int blockZ, int surfaceHeight)
{
    const int cellX = floordiv(blockX, 16);
    const int cellY = floordiv(blockY, 40);
    const int cellZ = floordiv(blockZ, 16);
    const int centerY = cellY*40 + 20;
    const double spread = sampleAquiferNoise(
        bn, AN_SPREAD, cellX, cellY, cellZ) * 10.0;
    const int offset = (int)floor(spread/3.0) * 3;
    const int fluidY = centerY + offset;
    return surfaceHeight < fluidY ? surfaceHeight : fluidY;
}

static AquiferFluidLevel calculateAquiferFluidLevel(
    const BiomeNoise *bn, int blockX, int blockY, int blockZ)
{
    static const int chunkOffsets[13][2] = {
        {0, 0}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1},
        {-3, 0}, {-2, 0}, {-1, 0}, {1, 0},
        {-2, 1}, {-1, 1}, {0, 1}, {1, 1},
    };
    const AquiferFluidLevel defaultLevel = aquiferGlobalFluid(blockY);
    int minimumSurface = INT32_MAX;
    const int upperY = blockY + 12;
    const int lowerY = blockY - 12;
    int foundSurfaceFluid = 0;
    int index;
    for (index = 0; index < 13; index++)
    {
        const int sampleX = blockX + chunkOffsets[index][0]*16;
        const int sampleZ = blockZ + chunkOffsets[index][1]*16;
        const int surface = estimateModernSurfaceHeight(bn, sampleX, sampleZ);
        const int surfaceWithMargin = (int32_t)((uint32_t)surface + 8U);
        const int center = index == 0;
        if (center && lowerY > surfaceWithMargin)
            return defaultLevel;
        const int aboveSurface = upperY > surfaceWithMargin;
        if (aboveSurface || center)
        {
            const AquiferFluidLevel surfaceLevel = aquiferGlobalFluid(
                surfaceWithMargin);
            if (aquiferFluidStateAt(&surfaceLevel, surfaceWithMargin)
                != AQUIFER_AIR)
            {
                if (center)
                    foundSurfaceFluid = 1;
                if (aboveSurface)
                    return surfaceLevel;
            }
        }
        if (surface < minimumSurface)
            minimumSurface = surface;
    }

    int fluidY;
    if (aquiferDeepDark(bn, blockX, blockY, blockZ))
    {
        fluidY = AQUIFER_NO_FLUID_Y;
    }
    else
    {
        const int distance = (int32_t)(
            (uint32_t)minimumSurface + 8U - (uint32_t)blockY);
        double surfacePressure = 0.0;
        if (foundSurfaceFluid)
        {
            surfacePressure = 1.0 - clampValue(distance/64.0, 0.0, 1.0);
        }
        const double floodedness = clampValue(sampleAquiferNoise(
            bn, AN_FLOODEDNESS, blockX, blockY, blockZ), -1.0, 1.0);
        const double upperThreshold = 0.8 - 1.1*surfacePressure;
        const double lowerThreshold = 0.4 - 1.2*surfacePressure;
        if (floodedness - upperThreshold > 0.0)
        {
            fluidY = defaultLevel.y;
        }
        else if (floodedness - lowerThreshold > 0.0)
        {
            fluidY = aquiferNoiseBasedFluidY(
                bn, blockX, blockY, blockZ, minimumSurface);
        }
        else
        {
            fluidY = AQUIFER_NO_FLUID_Y;
        }
    }
    int state = defaultLevel.state;
    if (fluidY <= -10 && fluidY != AQUIFER_NO_FLUID_Y
        && defaultLevel.state != AQUIFER_LAVA)
    {
        const int cellX = floordiv(blockX, 64);
        const int cellY = floordiv(blockY, 40);
        const int cellZ = floordiv(blockZ, 64);
        const double lava = sampleAquiferNoise(
            bn, AN_LAVA, cellX, cellY, cellZ);
        if (fabs(lava) > 0.3)
            state = AQUIFER_LAVA;
    }
    AquiferFluidLevel level = { fluidY, state };
    return level;
}

static uint64_t aquiferPositionHash(int x, int y, int z)
{
    const int32_t mixedX = (int32_t)((uint32_t)x * 3129871U);
    uint64_t hash = (uint64_t)(int64_t)mixedX
        ^ (uint64_t)(int64_t)z * 116129781ULL
        ^ (uint64_t)(int64_t)y;
    hash = hash*hash*42317861ULL + hash*11ULL;
    uint64_t shifted = hash >> 16;
    if (hash & (1ULL << 63))
        shifted |= 0xffff000000000000ULL;
    return shifted;
}

static AquiferCenter *loadAquiferCenter(
    AquiferCache *cache, int cellX, int cellY, int cellZ,
    AquiferCenter *scratch)
{
    AquiferCenter *center = scratch;
    if (cache->centers
        && cellX >= cache->minCellX && cellX < cache->minCellX + cache->sizeX
        && cellY >= cache->minCellY && cellY < cache->minCellY + cache->sizeY
        && cellZ >= cache->minCellZ && cellZ < cache->minCellZ + cache->sizeZ)
    {
        const int x = cellX - cache->minCellX;
        const int y = cellY - cache->minCellY;
        const int z = cellZ - cache->minCellZ;
        center = cache->centers + ((size_t)y*cache->sizeZ + z)*cache->sizeX + x;
    }
    if (!center->positionReady)
    {
        Xoroshiro random = {
            aquiferPositionHash(cellX, cellY, cellZ)
                ^ cache->bn->aquiferSeedLo,
            cache->bn->aquiferSeedHi,
        };
        center->blockX = cellX*16 + xNextInt(&random, 10);
        center->blockY = cellY*12 + xNextInt(&random, 9);
        center->blockZ = cellZ*16 + xNextInt(&random, 10);
        center->positionReady = 1;
    }
    if (!center->fluidReady)
    {
        center->fluid = calculateAquiferFluidLevel(cache->bn,
            center->blockX, center->blockY, center->blockZ);
        center->fluidReady = 1;
    }
    return center;
}

static double aquiferDistanceSimilarity(int first, int second)
{
    return 1.0 - fabs((double)second - first) / 25.0;
}

static double calculateAquiferBarrierDensity(
    const BiomeNoise *bn, int blockX, int blockY, int blockZ,
    const AquiferFluidLevel *first, const AquiferFluidLevel *second,
    double *barrier)
{
    const int firstState = aquiferFluidStateAt(first, blockY);
    const int secondState = aquiferFluidStateAt(second, blockY);
    if ((firstState == AQUIFER_LAVA && secondState == AQUIFER_WATER)
        || (firstState == AQUIFER_WATER && secondState == AQUIFER_LAVA))
        return 2.0;
    const int levelDistance = abs(first->y - second->y);
    if (levelDistance == 0)
        return 0.0;
    const double midpoint = 0.5 * (first->y + second->y);
    const double relativeY = blockY + 0.5 - midpoint;
    const double halfDistance = levelDistance / 2.0;
    const double remaining = halfDistance - fabs(relativeY);
    double pressure;
    if (relativeY > 0.0)
    {
        const double value = remaining;
        pressure = value > 0.0 ? value/1.5 : value/2.5;
    }
    else
    {
        const double value = 3.0 + remaining;
        pressure = value > 0.0 ? value/3.0 : value/10.0;
    }
    double barrierValue = 0.0;
    if (pressure >= -2.0 && pressure <= 2.0)
    {
        if (isnan(*barrier))
            *barrier = sampleAquiferNoise(
                bn, AN_BARRIER, blockX, blockY, blockZ);
        barrierValue = *barrier;
    }
    return 2.0 * (barrierValue + pressure);
}

static int sampleAquiferFluidState(
    AquiferCache *cache, int blockX, int blockY, int blockZ, double density)
{
    const AquiferFluidLevel globalFluid = aquiferGlobalFluid(blockY);
    if (aquiferFluidStateAt(&globalFluid, blockY) == AQUIFER_LAVA)
        return AQUIFER_LAVA;
    const int baseCellX = floordiv(blockX - 5, 16);
    const int baseCellY = floordiv(blockY + 1, 12);
    const int baseCellZ = floordiv(blockZ - 5, 16);
    int nearestDistance = INT32_MAX;
    int secondDistance = INT32_MAX;
    int thirdDistance = INT32_MAX;
    AquiferCenter nearest = {0};
    AquiferCenter second = {0};
    AquiferCenter third = {0};
    int x, y, z;
    for (x = 0; x <= 1; x++)
    {
        for (y = -1; y <= 1; y++)
        {
            for (z = 0; z <= 1; z++)
            {
                AquiferCenter scratch = {0};
                AquiferCenter *center = loadAquiferCenter(cache,
                    baseCellX+x, baseCellY+y, baseCellZ+z, &scratch);
                const int dx = center->blockX - blockX;
                const int dy = center->blockY - blockY;
                const int dz = center->blockZ - blockZ;
                const int distance = dx*dx + dy*dy + dz*dz;
                if (nearestDistance >= distance)
                {
                    third = second;
                    second = nearest;
                    nearest = *center;
                    thirdDistance = secondDistance;
                    secondDistance = nearestDistance;
                    nearestDistance = distance;
                }
                else if (secondDistance >= distance)
                {
                    third = second;
                    second = *center;
                    thirdDistance = secondDistance;
                    secondDistance = distance;
                }
                else if (thirdDistance >= distance)
                {
                    third = *center;
                    thirdDistance = distance;
                }
            }
        }
    }

    const double nearestSimilarity = aquiferDistanceSimilarity(
        nearestDistance, secondDistance);
    const int nearestState = aquiferFluidStateAt(&nearest.fluid, blockY);
    if (nearestSimilarity <= 0.0)
        return nearestState;
    double barrier = NAN;
    const double firstPressure = nearestSimilarity
        * calculateAquiferBarrierDensity(cache->bn,
            blockX, blockY, blockZ, &nearest.fluid, &second.fluid, &barrier);
    if (density + firstPressure > 0.0)
        return AQUIFER_AIR;
    const double thirdSimilarity = aquiferDistanceSimilarity(
        nearestDistance, thirdDistance);
    if (thirdSimilarity > 0.0)
    {
        const double pressure = nearestSimilarity * thirdSimilarity
            * calculateAquiferBarrierDensity(cache->bn,
                blockX, blockY, blockZ, &nearest.fluid, &third.fluid, &barrier);
        if (density + pressure > 0.0)
            return AQUIFER_AIR;
    }
    const double secondThirdSimilarity = aquiferDistanceSimilarity(
        secondDistance, thirdDistance);
    if (secondThirdSimilarity > 0.0)
    {
        const double pressure = nearestSimilarity * secondThirdSimilarity
            * calculateAquiferBarrierDensity(cache->bn,
                blockX, blockY, blockZ, &second.fluid, &third.fluid, &barrier);
        if (density + pressure > 0.0)
            return AQUIFER_AIR;
    }
    return nearestState;
}

static void initAquiferCache(
    AquiferCache *cache, const BiomeNoise *bn,
    int blockX, int blockZ, int w, int h, int stride)
{
    memset(cache, 0, sizeof(*cache));
    cache->bn = bn;
    const int lastX = (int)((int64_t)blockX + (int64_t)(w-1)*stride);
    const int lastZ = (int)((int64_t)blockZ + (int64_t)(h-1)*stride);
    cache->minCellX = floordiv(blockX - 5, 16);
    cache->minCellY = -7;
    cache->minCellZ = floordiv(blockZ - 5, 16);
    cache->sizeX = floordiv(lastX - 5, 16) + 2 - cache->minCellX;
    cache->sizeY = 14;
    cache->sizeZ = floordiv(lastZ - 5, 16) + 2 - cache->minCellZ;
    const size_t count = (size_t)cache->sizeX * cache->sizeY * cache->sizeZ;
    if (count <= AQUIFER_CACHE_LIMIT)
        cache->centers = (AquiferCenter*) calloc(count, sizeof(AquiferCenter));
}

static int findModernTerrainSurface(
    const BiomeNoise *bn,
    const ModernTerrainColumn *column00,
    const ModernTerrainColumn *column01,
    const ModernTerrainColumn *column10,
    const ModernTerrainColumn *column11,
    int cellX, int cellZ, double dx, double dz)
{
    int cellY;
    for (cellY = MODERN_MAX_CELL_Y - 1;
         cellY >= MODERN_MIN_CELL_Y; cellY--)
    {
        int y;
        for (y = 7; y >= 0; y--)
        {
            const double dy = y / 8.0;
            const double density = sampleModernBaseDensity(
                column00, column01, column10, column11,
                cellY, dx, dy, dz);
            if (density > 0.0 && sampleInterpolatedNoodle(
                    bn, cellX, cellY, cellZ, dx, dy, dz) > 0.0)
                return cellY * 8 + y;
        }
    }
    return -64;
}

static int findModernTerrainSurfaceUncached(
    const BiomeNoise *bn, int blockX, int blockZ)
{
    const int cellX = floordiv(blockX, 4);
    const int cellZ = floordiv(blockZ, 4);
    const double dx = (blockX - cellX*4) / 4.0;
    const double dz = (blockZ - cellZ*4) / 4.0;
    ModernTerrainColumn columns[4];
    sampleModernTerrainColumn(columns + 0, bn, cellX, cellZ);
    sampleModernTerrainColumn(columns + 1, bn, cellX, cellZ+1);
    sampleModernTerrainColumn(columns + 2, bn, cellX+1, cellZ);
    sampleModernTerrainColumn(columns + 3, bn, cellX+1, cellZ+1);
    return findModernTerrainSurface(bn,
        columns + 0, columns + 1, columns + 2, columns + 3,
        cellX, cellZ, dx, dz);
}

static int resolveModernFluidSurface(
    AquiferCache *aquifer,
    const ModernTerrainColumn *column00,
    const ModernTerrainColumn *column01,
    const ModernTerrainColumn *column10,
    const ModernTerrainColumn *column11,
    int blockX, int blockZ, int solidY)
{
    const int cellX = floordiv(blockX, 4);
    const int cellZ = floordiv(blockZ, 4);
    const double dx = (blockX - cellX*4) / 4.0;
    const double dz = (blockZ - cellZ*4) / 4.0;
    int blockY;
    for (blockY = 62; blockY > solidY; blockY--)
    {
        const int cellY = floordiv(blockY, 8);
        const double dy = (blockY - cellY*8) / 8.0;
        const double density = sampleModernFinalDensity(aquifer->bn,
            column00, column01, column10, column11,
            cellX, cellY, cellZ, dx, dy, dz);
        const int state = sampleAquiferFluidState(
            aquifer, blockX, blockY, blockZ, density);
        if (state == AQUIFER_WATER)
            return blockY;
        if (state == AQUIFER_LAVA)
            return INT32_MIN;
    }
    return INT32_MIN;
}

static int mapOverworldTerrainHeightBlocksImpl(
    int *y, int *fluidY, const BiomeNoise *bn,
    int blockX, int blockZ, int w, int h, int stride)
{
    if (!y || !bn || w <= 0 || h <= 0 || stride <= 0)
        return 1;
    const int lastX = (int)((int64_t)blockX + (int64_t)(w-1)*stride);
    const int lastZ = (int)((int64_t)blockZ + (int64_t)(h-1)*stride);
    const int minCellX = floordiv(blockX, 4);
    const int minCellZ = floordiv(blockZ, 4);
    const int maxCellX = floordiv(lastX, 4) + 1;
    const int maxCellZ = floordiv(lastZ, 4) + 1;
    const int cellW = maxCellX - minCellX + 1;
    const int cellH = maxCellZ - minCellZ + 1;
    const size_t cellCount = (size_t)cellW * cellH;
    ModernTerrainColumn *columns = NULL;
    if (cellCount <= MODERN_COLUMN_CACHE_LIMIT)
        columns = (ModernTerrainColumn*) malloc(
            sizeof(ModernTerrainColumn) * cellCount);

    int i, j;
    if (columns)
    {
        for (j = 0; j < cellH; j++)
            for (i = 0; i < cellW; i++)
                sampleModernTerrainColumn(columns + (size_t)j*cellW+i,
                    bn, minCellX+i, minCellZ+j);
    }
    AquiferCache aquifer;
    if (fluidY)
        initAquiferCache(&aquifer, bn, blockX, blockZ, w, h, stride);

    for (j = 0; j < h; j++)
    {
        const int sampleZ = (int)((int64_t)blockZ + (int64_t)j*stride);
        const int cellZ = floordiv(sampleZ, 4);
        const double dz = (sampleZ - cellZ*4) / 4.0;
        for (i = 0; i < w; i++)
        {
            const int sampleX = (int)((int64_t)blockX + (int64_t)i*stride);
            if (!columns)
            {
                y[j*w+i] = findModernTerrainSurfaceUncached(
                    bn, sampleX, sampleZ);
                if (fluidY)
                {
                    if (y[j*w+i] >= 62)
                    {
                        fluidY[j*w+i] = INT32_MIN;
                    }
                    else
                    {
                        ModernTerrainColumn local[4];
                        const int localCellX = floordiv(sampleX, 4);
                        sampleModernTerrainColumn(local + 0, bn, localCellX, cellZ);
                        sampleModernTerrainColumn(local + 1, bn, localCellX, cellZ+1);
                        sampleModernTerrainColumn(local + 2, bn, localCellX+1, cellZ);
                        sampleModernTerrainColumn(local + 3, bn, localCellX+1, cellZ+1);
                        fluidY[j*w+i] = resolveModernFluidSurface(
                            &aquifer, local + 0, local + 1, local + 2, local + 3,
                            sampleX, sampleZ, y[j*w+i]);
                    }
                }
                continue;
            }
            const int cellX = floordiv(sampleX, 4);
            const double dx = (sampleX - cellX*4) / 4.0;
            const int offsetX = cellX - minCellX;
            const int offsetZ = cellZ - minCellZ;
            const ModernTerrainColumn *column00 = columns
                + (size_t)offsetZ*cellW + offsetX;
            const ModernTerrainColumn *column01 = column00 + cellW;
            y[j*w+i] = findModernTerrainSurface(bn,
                column00, column01, column00+1, column01+1,
                cellX, cellZ, dx, dz);
            if (fluidY)
            {
                fluidY[j*w+i] = y[j*w+i] >= 62
                    ? INT32_MIN
                    : resolveModernFluidSurface(&aquifer,
                        column00, column01, column00+1, column01+1,
                        sampleX, sampleZ, y[j*w+i]);
            }
        }
    }
    if (fluidY)
        free(aquifer.centers);
    free(columns);
    return 0;
}

int mapOverworldTerrainHeightBlocks(
    int *y, const BiomeNoise *bn,
    int blockX, int blockZ, int w, int h, int stride)
{
    return mapOverworldTerrainHeightBlocksImpl(
        y, NULL, bn, blockX, blockZ, w, h, stride);
}

int mapOverworldSurfaceHeightBlocks(
    int *solidY, int *fluidY, const BiomeNoise *bn,
    int blockX, int blockZ, int w, int h, int stride)
{
    if (!fluidY)
        return 1;
    return mapOverworldTerrainHeightBlocksImpl(
        solidY, fluidY, bn, blockX, blockZ, w, h, stride);
}

int getOverworldTerrainHeight(
    const BiomeNoise *bn, int blockX, int blockZ)
{
    return findModernTerrainSurfaceUncached(bn, blockX, blockZ);
}

int mapOverworldTerrainHeight(float *y, int *ids, const BiomeNoise *bn,
    int x, int z, int w, int h)
{
    int i, j;
    for (j = 0; j < h; j++)
    {
        for (i = 0; i < w; i++)
        {
            const int px = x + i;
            const int pz = z + j;
            const int surface = findModernTerrainSurfaceUncached(
                bn, px * 4, pz * 4);
            if (ids)
            {
                ids[j*w+i] = sampleBiomeNoise(
                    bn, 0, px, floordiv(surface > 62 ? surface : 62, 4), pz, 0, 0);
            }
            y[j*w+i] = surface;
        }
    }
    return 0;
}

int mapOverworldSurfaceBiome(int *out, const BiomeNoise *bn, uint64_t sha,
    int scale, int x, int z, int w, int h)
{
    if (scale != 1 && scale != 4)
        return 1;

    if (scale == 4)
    {
        float *height = malloc(sizeof(float) * (size_t)w * h);
        if (!height)
            return 1;
        int err = mapOverworldTerrainHeight(height, 0, bn, x, z, w, h);
        if (!err)
        {
            int i, j;
            for (j = 0; j < h; j++)
            {
                for (i = 0; i < w; i++)
                {
                    int surface = (int)height[j*w+i];
                    if (surface < 62) surface = 62;
                    out[j*w+i] = sampleBiomeNoise(bn, 0,
                        x+i, floordiv(surface, 4), z+j, 0, 0);
                }
            }
        }
        free(height);
        return err;
    }

    const int x4min = floordiv(x, 4);
    const int z4min = floordiv(z, 4);
    const int x4max = floordiv(x + w - 1, 4) + 1;
    const int z4max = floordiv(z + h - 1, 4) + 1;
    const int sw = x4max - x4min + 1;
    const int sh = z4max - z4min + 1;
    float *height = malloc(sizeof(float) * (size_t)sw * sh);
    if (!height)
        return 1;
    int err = mapOverworldTerrainHeight(
        height, 0, bn, x4min, z4min, sw, sh);
    if (err)
    {
        free(height);
        return err;
    }

    int i, j;
    for (j = 0; j < h; j++)
    {
        const int blockZ = z + j;
        const int baseZ = floordiv(blockZ, 4);
        const int fz = blockZ - baseZ * 4;
        const int iz = baseZ - z4min;
        for (i = 0; i < w; i++)
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
            out[j*w+i] = sampleBiomeNoise(bn, 0, x4, y4, z4, 0, 0);
        }
    }
    free(height);
    return 0;
}

void sampleNoiseParameters(BiomeNoise *bn, int x, int z, float np_param[4]) {
    double px = x + sampleDoublePerlin(&bn->climate[NP_SHIFT], x, 0, z) * 4.0;
    double pz = z + sampleDoublePerlin(&bn->climate[NP_SHIFT], z, x, 0) * 4.0;

    float c = (float)sampleDoublePerlin(&bn->climate[NP_CONTINENTALNESS], px, 0, pz);
    float e = (float)sampleDoublePerlin(&bn->climate[NP_EROSION], px, 0, pz);
    float w = (float)sampleDoublePerlin(&bn->climate[NP_WEIRDNESS], px, 0, pz);

    np_param[0] = c;
    np_param[1] = e;
    np_param[2] = peaksAndValleys(w);
    np_param[3] = w;
}

// Note: Climate noise is sampled at a 1:1 scale.
int sampleBiomeNoiseBeta(const BiomeNoiseBeta *bnb, int64_t *np, double *nv,
    int x, int z)
{
    if (bnb->nptype >= 0 && np)
        memset(np, 0, 2*sizeof(*np));
    double t, h, f;
    f = sampleOctaveBeta17Biome(&bnb->climate[2], x, z) * 1.1 + 0.5;

    t = (sampleOctaveBeta17Biome(&bnb->climate[0], x, z) *
        0.15 + 0.7) * 0.99 + f * 0.01;
    t = 1 - (1 - t) * (1 - t);
    t = (t < 0) ? 0 : t;
    t = (t > 1) ? 1 : t;
    if (bnb->nptype == NP_TEMPERATURE)
        return (int64_t) (10000.0F * t);

    h = (sampleOctaveBeta17Biome(&bnb->climate[1], x, z) *
        0.15 + 0.5) * 0.998 + f * 0.002;
    h = (h < 0) ? 0 : h;
    h = (h > 1) ? 1 : h;
    if (bnb->nptype == NP_HUMIDITY)
        return (int64_t) (10000.0F * h * t);

    if (nv)
    {
        nv[0] = t;
        nv[1] = h;
    }
    return getOldBetaBiome((float) t, (float) h);
}


int getOldBetaBiome(float t, float h)
{
    static const uint8_t biome_table_beta_1_7[64*64] =
    {
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,
        9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,
        9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,
        9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,0,0,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,
        9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,
        9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,
        9,9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,
        9,9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,
        9,9,9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,
        9,9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,
        9,9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,
        9,9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,
        9,9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,
        9,9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,
        9,9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,
        9,9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,
        9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,
        9,9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,
        9,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,2,2,2,2,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,2,2,2,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,2,2,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,2,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,7,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,7,8,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,8,8,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,8,8,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,8,8,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,8,8,
        5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,8,8,
        5,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,8,8,
    };
    static const int bmap[] = {
        plains, desert, forest, taiga, swamp, snowy_tundra, savanna,
        seasonal_forest, rainforest, shrubland
    };
    int idx = (int)(t * 63) + (int)(h * 63) * 64;
    return bmap[ biome_table_beta_1_7[idx] ];
}


static
uint64_t get_np_dist(const uint64_t np[6], const BiomeTree *bt, int idx)
{
    uint64_t ds = 0, node = bt->nodes[idx];
    uint64_t a, b, d;
    uint32_t i;

    for (i = 0; i < 6; i++)
    {
        idx = (node >> 8*i) & 0xFF;
        a = np[i] - bt->param[2*idx + 1];
        b = bt->param[2*idx + 0] - np[i];
        d = (int64_t)a > 0 ? a : (int64_t)b > 0 ? b : 0;
        d = d * d;
        ds += d;
    }
    return ds;
}

static
int get_resulting_node(const uint64_t np[6], const BiomeTree *bt, int idx,
    int alt, uint64_t ds, int depth)
{
    if (bt->steps[depth] == 0)
        return idx;
    uint32_t step;
    do
    {
        step = bt->steps[depth];
        depth++;
    }
    while (idx+step >= bt->len);

    uint64_t node = bt->nodes[idx];
    uint16_t inner = node >> 48;

    int leaf = alt;
    uint32_t i, n;

    for (i = 0, n = bt->order; i < n; i++)
    {
        uint64_t ds_inner = get_np_dist(np, bt, inner);
        if (ds_inner < ds)
        {
            int leaf2 = get_resulting_node(np, bt, inner, leaf, ds, depth);
            uint64_t ds_leaf2;
            if (inner == leaf2)
                ds_leaf2 = ds_inner;
            else
                ds_leaf2 = get_np_dist(np, bt, leaf2);
            if (ds_leaf2 < ds)
            {
                ds = ds_leaf2;
                leaf = leaf2;
            }
        }

        inner += step;
        if (inner >= bt->len)
            break;
    }

    return leaf;
}

ATTR(hot, flatten)
int climateToBiome(int mc, const uint64_t np[6], uint64_t *dat)
{
    static const BiomeTree btree18 = {
        btree18_steps, &btree18_param[0][0], btree18_nodes, btree18_order,
        sizeof(btree18_nodes) / sizeof(uint64_t)
    };
    static const BiomeTree btree192 = {
        btree192_steps, &btree192_param[0][0], btree192_nodes, btree192_order,
        sizeof(btree192_nodes) / sizeof(uint64_t)
    };
    static const BiomeTree btree19 = {
        btree19_steps, &btree19_param[0][0], btree19_nodes, btree19_order,
        sizeof(btree19_nodes) / sizeof(uint64_t)
    };
    static const BiomeTree btree20 = {
        btree20_steps, &btree20_param[0][0], btree20_nodes, btree20_order,
        sizeof(btree20_nodes) / sizeof(uint64_t)
    };
    static const BiomeTree btree21wd = {
        btree21wd_steps, &btree21wd_param[0][0], btree21wd_nodes, btree21wd_order,
        sizeof(btree21wd_nodes) / sizeof(uint64_t)
    };
    static const BiomeTree btree215 = {
        btree215_steps, &btree215_param[0][0], btree215_nodes, btree215_order,
        sizeof(btree215_nodes) / sizeof(uint64_t)
    };
    static const BiomeTree btree262 = {
        btree262_steps, &btree262_param[0][0], btree262_nodes, btree262_order,
        sizeof(btree262_nodes) / sizeof(uint64_t)
    };
    static const BiomeTree btree263 = {
        btree263_steps, &btree263_param[0][0], btree263_nodes, btree263_order,
        sizeof(btree263_nodes) / sizeof(uint64_t)
    };

    const BiomeTree *bt;
    int idx;
    // When adding a biome tree for a newer version, do not forget the 'else'
    // on the branch that used to be first.
    if (mc >= MC_26_3)
        bt = &btree263;
    else if (mc >= MC_26_2)
        bt = &btree262;
    else if (mc >= MC_1_21_5)
        bt = &btree215;
    else if (mc >= MC_1_21_4)
        bt = &btree21wd;
    else if (mc >= MC_1_20_6)
        bt = &btree20;
    else if (mc >= MC_1_19_4)
        bt = &btree19;
    else if (mc >= MC_1_19_2)
        bt = &btree192;
    else
        bt = &btree18;

    if (dat)
    {
        int alt = (int) *dat;
        uint64_t ds = get_np_dist(np, bt, alt);
        idx = get_resulting_node(np, bt, 0, alt, ds, 0);
        *dat = (uint64_t) idx;
    }
    else
    {
        idx = get_resulting_node(np, bt, 0, 0, -1, 0);
    }

    return (bt->nodes[idx] >> 48) & 0xFF;
}


void setClimateParaSeed(BiomeNoise *bn, uint64_t seed, int large, int nptype, int nmax)
{
    Xoroshiro pxr;
    xSetSeed(&pxr, seed);
    uint64_t xlo = xNextLong(&pxr);
    uint64_t xhi = xNextLong(&pxr);
    if (nptype == NP_DEPTH)
    {
        int n = 0;
        n += init_climate_seed(bn->climate + NP_CONTINENTALNESS,
            bn->oct + n, xlo, xhi, large,    NP_CONTINENTALNESS, nmax);
        n += init_climate_seed(bn->climate + NP_EROSION,
            bn->oct + n, xlo, xhi, large,    NP_EROSION, nmax);
        n += init_climate_seed(bn->climate + NP_WEIRDNESS,
            bn->oct + n, xlo, xhi, large,    NP_WEIRDNESS, nmax);
    }
    else
    {
        init_climate_seed(bn->climate + nptype, bn->oct, xlo, xhi, large, nptype, nmax);
    }
    bn->nptype = nptype;
}

double sampleClimatePara(const BiomeNoise *bn, int64_t *np, double x, double z)
{
    if (bn->nptype == NP_DEPTH)
    {
        float c, e, w;
        c = sampleDoublePerlin(bn->climate + NP_CONTINENTALNESS, x, 0, z);
        e = sampleDoublePerlin(bn->climate + NP_EROSION, x, 0, z);
        w = sampleDoublePerlin(bn->climate + NP_WEIRDNESS, x, 0, z);

        float np_param[] = {
            c, e, peaksAndValleys(w), w,
        };
        double off = getSpline(bn->sp, np_param) + 0.015F;
        int y = 0;
        float d = 1.0 - (y * 4) / 128.0 - 83.0/160.0 + off;
        if (np)
        {
            np[2] = (int64_t)(10000.0F*c);
            np[3] = (int64_t)(10000.0F*e);
            np[4] = (int64_t)(10000.0F*d);
            np[5] = (int64_t)(10000.0F*w);
        }
        return d;
    }
    double p = sampleDoublePerlin(bn->climate + bn->nptype, x, 0, z);
    if (np)
        np[bn->nptype] = (int64_t)(10000.0F*p);
    return p;
}

void genBiomeNoiseChunkSection(const BiomeNoise *bn, int out[4][4][4],
    int cx, int cy, int cz, uint64_t *dat)
{
    uint64_t buf = 0;
    int i, j, k;
    int x4 = cx * 4, y4 = cy * 4, z4 = cz * 4;
    if (dat == NULL)
        dat = &buf;
    if (*dat == 0)
    {   // try to determine the ending point of the last chunk section
        sampleBiomeNoise(bn, NULL, x4+3, y4-1, z4+3, dat, 0);
    }

    // iteration order is important
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            for (k = 0; k < 4; ++k) {
                out[i][j][k] = sampleBiomeNoise(bn, NULL, x4+i, y4+j, z4+k, dat, 0);
            }
        }
    }
}

static void genBiomeNoise3D(const BiomeNoise *bn, int *out, Range r, int opt)
{
    uint64_t dat = 0;
    uint64_t *p_dat = opt ? &dat : NULL;
    uint32_t flags = opt ? SAMPLE_NO_SHIFT : 0;
    int i, j, k;
    int *p = out;
    int scale = r.scale > 4 ? r.scale / 4 : 1;
    int mid = scale / 2;
    for (k = 0; k < r.sy; k++)
    {
        int yk = (r.y+k);
        for (j = 0; j < r.sz; j++)
        {
            int zj = (r.z+j)*scale + mid;
            for (i = 0; i < r.sx; i++)
            {
                int xi = (r.x+i)*scale + mid;
                *p = sampleBiomeNoise(bn, NULL, xi, yk, zj, p_dat, flags);
                p++;
            }
        }
    }
}

int genBiomeNoiseScaled(const BiomeNoise *bn, int *out, Range r, uint64_t sha)
{
    if (r.sy == 0)
        r.sy = 1;

    uint64_t siz = (uint64_t)r.sx*r.sy*r.sz;
    int i, j, k;

    if (r.scale == 1)
    {
        Range s = getVoronoiSrcRange(r);
        int *src;
        if (siz > 1)
        {   // the source range is large enough that we can try optimizing
            src = out + siz;
            genBiomeNoise3D(bn, src, s, 0);
        }
        else
        {
            src = NULL;
        }

        int *p = out;
        for (k = 0; k < r.sy; k++)
        {
            for (j = 0; j < r.sz; j++)
            {
                for (i = 0; i < r.sx; i++)
                {
                    int x4, z4, y4;
                    voronoiAccess3D(sha, r.x+i, r.y+k, r.z+j, &x4, &y4, &z4);
                    if (src)
                    {
                        x4 -= s.x; y4 -= s.y; z4 -= s.z;
                        *p = src[(int64_t)y4*s.sx*s.sz + (int64_t)z4*s.sx + x4];
                    }
                    else
                    {
                        *p = sampleBiomeNoise(bn, 0, x4, y4, z4, 0, 0);
                    }
                    p++;
                }
            }
        }
    }
    else
    {
        // There is (was?) an optimization that causes MC-241546, and should
        // not be enabled for accurate results. However, if the scale is higher
        // than 1:4, the accuracy becomes questionable anyway. Furthermore
        // situations that want to use a higher scale are usually better off
        // with a faster, if imperfect, result.
        genBiomeNoise3D(bn, out, r, r.scale > 4);
    }
    return 0;
}

static void genColumnNoise(const SurfaceNoiseBeta *snb, SeaLevelColumnNoiseBeta *dest,
    double cx, double cz, double lacmin)
{
    dest->contASample = sampleOctaveAmp(&snb->octcontA, cx, 0, cz, 0, 0, 1);
    dest->contBSample = sampleOctaveAmp(&snb->octcontB, cx, 0, cz, 0, 0, 1);
    sampleOctaveBeta17Terrain(&snb->octmin, dest->minSample, cx, cz, 0, lacmin);
    sampleOctaveBeta17Terrain(&snb->octmax, dest->maxSample, cx, cz, 0, lacmin);
    sampleOctaveBeta17Terrain(&snb->octmain, dest->mainSample, cx, cz, 1, lacmin);
}

static void processColumnNoise(double *out, const SeaLevelColumnNoiseBeta *src,
    const double climate[2])
{
    double humi = 1 - climate[0] * climate[1];
    humi *= humi;
    humi *= humi;
    humi = 1 - humi;
    double contA = (src->contASample + 256) / 512 * humi;
    contA = (contA > 1) ? 1.0 : contA;
    double contB = src->contBSample / 8000;
    if (contB < 0)
        contB = -contB * 0.3;
    contB = contB*3-2;
    if (contB < 0)
    {
        contB /= 2;
        contB = (contB < -1) ? -1.0 / 1.4 / 2 : contB / 1.4 / 2;
        contA = 0;
    }
    else
    {
        contB = (contB > 1) ? 1.0/8 : contB/8;
    }
    contA = (contA < 0) ? 0.5 : contA+0.5;
    contB = (contB * 17.0) / 16;
    contB = 17.0 / 2 + contB * 4;
    const double *low = src->minSample;
    const double *high = src->maxSample;
    const double *selector = src->mainSample;
    int i;
    for (i = 0; i <= 1; i++)
    {
        double chooseLHS;
        double procCont = ((i + 7 - contB) * 12) / contA;
        procCont = (procCont < 0) ? procCont*4 : procCont;
        double lSample = low[i] / 512;
        double hSample = high[i] / 512;
        double sSample = (selector[i] / 10 + 1) / 2;
        chooseLHS = (sSample < 0.0) ? lSample : (sSample > 1) ? hSample :
            lSample + (hSample - lSample) * sSample;
        chooseLHS -= procCont;
        out[i] = chooseLHS;
    }
}

static double lerp4(
    const double a[2], const double b[2], const double c[2], const double d[2],
    double dy, double dx, double dz)
{
    double b00 = a[0] + (a[1] - a[0]) * dy;
    double b01 = b[0] + (b[1] - b[0]) * dy;
    double b10 = c[0] + (c[1] - c[0]) * dy;
    double b11 = d[0] + (d[1] - d[0]) * dy;
    double b0 = b00 + (b10 - b00) * dz;
    double b1 = b01 + (b11 - b01) * dz;
    return b0 + (b1 - b0) * dx;
}

double approxSurfaceBeta(const BiomeNoiseBeta *bnb, const SurfaceNoiseBeta *snb,
    int x, int z)
{
    // TODO: sample vertically to get a more accurate height value
    double climate[2];
    sampleBiomeNoiseBeta(bnb, NULL, climate, x, z);
    double cols[2];
    SeaLevelColumnNoiseBeta colNoise;
    genColumnNoise(snb, &colNoise, x*0.25, z*0.25, 0);
    processColumnNoise(cols, &colNoise, climate);
    return 63 + (cols[0]*0.125 + cols[1]*0.875) * 0.5;
}

int genBiomeNoiseBetaScaled(const BiomeNoiseBeta *bnb,
    const SurfaceNoiseBeta *snb, int *out, Range r)
{
    if (!snb || r.scale >= 4)
    {
        int i, j;
        int mid = r.scale >> 1;
        for (j = 0; j < r.sz; j++)
        {
            int z = (r.z+j)*r.scale + mid;
            for (i = 0; i < r.sx; i++)
            {
                double climate[2];
                int x = (r.x+i)*r.scale + mid;
                int id = sampleBiomeNoiseBeta(bnb, NULL, climate, x, z);

                if (snb)
                {
                    double cols[2];
                    SeaLevelColumnNoiseBeta colNoise;
                    genColumnNoise(snb, &colNoise, x*0.25, z*0.25, 4.0/r.scale);
                    processColumnNoise(cols, &colNoise, climate);
                    if (cols[0]*0.125 + cols[1]*0.875 <= 0)
                        id = (climate[0] < 0.5) ? frozen_ocean : ocean;
                }
                out[(int64_t)j*r.sx + i] = id;
            }
        }
        return 0;
    }

    int cellwidth = r.scale >> 1;
    int cx1 = r.x >> (2 >> cellwidth);
    int cz1 = r.z >> (2 >> cellwidth);
    int cx2 = cx1 + (r.sx >> (2 >> cellwidth)) + 1;
    int cz2 = cz1 + (r.sz >> (2 >> cellwidth)) + 1;
    int steps = 4 >> cellwidth;
    int minDim, maxDim;
    if (cx2-cx1 > cz2-cz1) {
        maxDim = cx2-cx1;
        minDim = cz2-cz1;
    } else {
        maxDim = cz2-cz1;
        minDim = cx2-cx1;
    }
    int bufLen = minDim * 2 + 1;

    int i, j, x, z, cx, cz;
    int xStart = cx1;
    int zStart = cz1;
    int idx = 0;
    SeaLevelColumnNoiseBeta *buf = (SeaLevelColumnNoiseBeta*) (out + (int64_t)r.sx * r.sz);
    SeaLevelColumnNoiseBeta *colNoise;
    double cols[8];
    double climate[2];
    static const int off[] = { 1, 4, 7, 10, 13 };

    // Diagonal traversal of range region, in order to minimize size of saved
    // column noise buffer
    int stripe;
    for (stripe = 0; stripe < maxDim + minDim - 1; stripe++)
    {
        cx = xStart;
        cz = zStart;
        while (cx < cx2 && cz >= cz1)
        {
            int csx = (cx * 4) & ~15; // start of chunk coordinates
            int csz = (cz * 4) & ~15;
            int ci = cx & 3;
            int cj = cz & 3;

            colNoise = &buf[idx];
            if (stripe == 0)
                genColumnNoise(snb, colNoise, cx, cz, 0);
            sampleBiomeNoiseBeta(bnb, NULL, climate, csx+off[ci], csz+off[cj]);
            processColumnNoise(&cols[0], colNoise, climate);

            colNoise = &buf[(idx + minDim + 1) % bufLen];
            if (cz == cz1)
                genColumnNoise(snb, colNoise, cx+1, cz, 0);
            sampleBiomeNoiseBeta(bnb, NULL, climate, csx+off[ci+1], csz+off[cj]);
            processColumnNoise(&cols[2], colNoise, climate);

            colNoise = &buf[(idx + minDim) % bufLen];
            if (cx == cx1)
                genColumnNoise(snb, colNoise, cx, cz+1, 0);
            sampleBiomeNoiseBeta(bnb, NULL, climate, csx+off[ci], csz+off[cj+1]);
            processColumnNoise(&cols[4], colNoise, climate);

            colNoise = &buf[idx];
            genColumnNoise(snb, colNoise, cx+1, cz+1, 0);
            sampleBiomeNoiseBeta(bnb, NULL, climate, csx+off[ci+1], csz+off[cj+1]);
            processColumnNoise(&cols[6], colNoise, climate);

            // scale=1: cellwidth=0, steps=4
            // scale=4: cellwidth=2, steps=1
            for (j = 0; j < steps; j++)
            {
                z = cz * steps + j;
                if (z < r.z || z >= r.z + r.sz)
                    continue;
                for (i = 0; i < steps; i++)
                {
                    x = cx * steps + i;
                    if (x < r.x || x >= r.x + r.sx)
                        continue;
                    int mid = r.scale >> 1;
                    int bx = x * r.scale + mid;
                    int bz = z * r.scale + mid;
                    int id = sampleBiomeNoiseBeta(bnb, NULL, climate, bx, bz);
                    double dx = (bx & 3) * 0.25;
                    double dz = (bz & 3) * 0.25;
                    if (lerp4(cols+0, cols+2, cols+4, cols+6, 7./8, dx, dz) <= 0)
                        id = (climate[0] < 0.5) ? frozen_ocean : ocean;
                    out[(int64_t)(z - r.z) * r.sx + (x - r.x)] = id;
                }
            }

            cx++;
            cz--;
            idx = (idx+1) % bufLen;
        }
        if (zStart < cz2-1)
            zStart++;
        else
            xStart++;
        if (stripe+1 < minDim)
            idx = (idx + minDim-stripe-1) % bufLen;
        else if (stripe+1 > maxDim)
            idx = (idx + stripe-maxDim+2) % bufLen;
        else if (xStart > cx1)
            idx = (idx + 1) % bufLen;
    }
    return 0;
}


int getBiomeDepthAndScale(int id, double *depth, double *scale, int *grass)
{
    const int dh = 62; // default height
    double s = 0, d = 0, g = 0;
    switch (id) {
    case ocean:                         s = 0.100; d = -1.000; g = dh; break;
    case plains:                        s = 0.050; d =  0.125; g = dh; break;
    case desert:                        s = 0.050; d =  0.125; g =  0; break;
    case mountains:                     s = 0.500; d =  1.000; g = dh; break;
    case forest:                        s = 0.200; d =  0.100; g = dh; break;
    case taiga:                         s = 0.200; d =  0.200; g = dh; break;
    case swamp:                         s = 0.100; d = -0.200; g = dh; break;
    case river:                         s = 0.000; d = -0.500; g = 60; break;
    case frozen_ocean:                  s = 0.100; d = -1.000; g = dh; break;
    case frozen_river:                  s = 0.000; d = -0.500; g = 60; break;
    case snowy_tundra:                  s = 0.050; d =  0.125; g = dh; break;
    case snowy_mountains:               s = 0.300; d =  0.450; g = dh; break;
    case mushroom_fields:               s = 0.300; d =  0.200; g =  0; break;
    case mushroom_field_shore:          s = 0.025; d =  0.000; g =  0; break;
    case beach:                         s = 0.025; d =  0.000; g = 64; break;
    case desert_hills:                  s = 0.300; d =  0.450; g =  0; break;
    case wooded_hills:                  s = 0.300; d =  0.450; g = dh; break;
    case taiga_hills:                   s = 0.300; d =  0.450; g = dh; break;
    case mountain_edge:                 s = 0.300; d =  0.800; g = dh; break;
    case jungle:                        s = 0.200; d =  0.100; g = dh; break;
    case jungle_hills:                  s = 0.300; d =  0.450; g = dh; break;
    case jungle_edge:                   s = 0.200; d =  0.100; g = dh; break;
    case deep_ocean:                    s = 0.100; d = -1.800; g = dh; break;
    case stone_shore:                   s = 0.800; d =  0.100; g = 64; break;
    case snowy_beach:                   s = 0.025; d =  0.000; g = 64; break;
    case birch_forest:                  s = 0.200; d =  0.100; g = dh; break;
    case birch_forest_hills:            s = 0.300; d =  0.450; g = dh; break;
    case dark_forest:                   s = 0.200; d =  0.100; g = dh; break;
    case snowy_taiga:                   s = 0.200; d =  0.200; g = dh; break;
    case snowy_taiga_hills:             s = 0.300; d =  0.450; g = dh; break;
    case giant_tree_taiga:              s = 0.200; d =  0.200; g = dh; break;
    case giant_tree_taiga_hills:        s = 0.300; d =  0.450; g = dh; break;
    case wooded_mountains:              s = 0.500; d =  1.000; g = dh; break;
    case savanna:                       s = 0.050; d =  0.125; g = dh; break;
    case savanna_plateau:               s = 0.025; d =  1.500; g = dh; break;
    case badlands:                      s = 0.200; d =  0.100; g =  0; break;
    case wooded_badlands_plateau:       s = 0.025; d =  1.500; g =  0; break;
    case badlands_plateau:              s = 0.025; d =  1.500; g =  0; break;
    case warm_ocean:                    s = 0.100; d = -1.000; g =  0; break;
    case lukewarm_ocean:                s = 0.100; d = -1.000; g = dh; break;
    case cold_ocean:                    s = 0.100; d = -1.000; g = dh; break;
    case deep_warm_ocean:               s = 0.100; d = -1.800; g =  0; break;
    case deep_lukewarm_ocean:           s = 0.100; d = -1.800; g = dh; break;
    case deep_cold_ocean:               s = 0.100; d = -1.800; g = dh; break;
    case deep_frozen_ocean:             s = 0.100; d = -1.800; g = dh; break;
    case sunflower_plains:              s = 0.050; d =  0.125; g = dh; break;
    case desert_lakes:                  s = 0.250; d =  0.225; g =  0; break;
    case gravelly_mountains:            s = 0.500; d =  1.000; g = dh; break;
    case flower_forest:                 s = 0.400; d =  0.100; g = dh; break;
    case taiga_mountains:               s = 0.400; d =  0.300; g = dh; break;
    case swamp_hills:                   s = 0.300; d = -0.100; g = dh; break;
    case ice_spikes:                    s = 0.450; d =  0.425; g =  0; break;
    case modified_jungle:               s = 0.400; d =  0.200; g = dh; break;
    case modified_jungle_edge:          s = 0.400; d =  0.200; g = dh; break;
    case tall_birch_forest:             s = 0.400; d =  0.200; g = dh; break;
    case tall_birch_hills:              s = 0.500; d =  0.550; g = dh; break;
    case dark_forest_hills:             s = 0.400; d =  0.200; g = dh; break;
    case snowy_taiga_mountains:         s = 0.400; d =  0.300; g = dh; break;
    case giant_spruce_taiga:            s = 0.200; d =  0.200; g = dh; break;
    case giant_spruce_taiga_hills:      s = 0.200; d =  0.200; g = dh; break;
    case modified_gravelly_mountains:   s = 0.500; d =  1.000; g = dh; break;
    case shattered_savanna:             s = 1.225; d = 0.3625; g = dh; break;
    case shattered_savanna_plateau:     s = 1.212; d =  1.050; g = dh; break;
    case eroded_badlands:               s = 0.200; d =  0.100; g =  0; break;
    case modified_wooded_badlands_plateau: s = 0.300; d = 0.450; g = 0; break;
    case modified_badlands_plateau:     s = 0.300; d =  0.450; g =  0; break;
    case bamboo_jungle:                 s = 0.200; d =  0.100; g = dh; break;
    case bamboo_jungle_hills:           s = 0.300; d =  0.450; g = dh; break;
    default:
        return 0;
    }
    if (scale) *scale = s;
    if (depth) *depth = d;
    if (grass) *grass = g;
    return 1;
}


Range getVoronoiSrcRange(Range r)
{
    if (r.scale != 1)
    {
        printf("getVoronoiSrcRange() expects input range with scale 1:1\n");
        exit(1);
    }

    Range s; // output has scale 1:4
    int x = r.x - 2;
    int z = r.z - 2;
    s.scale = 4;
    s.x = x >> 2;
    s.z = z >> 2;
    s.sx = ((x + r.sx) >> 2) - s.x + 2;
    s.sz = ((z + r.sz) >> 2) - s.z + 2;
    if (r.sy < 1)
    {
        s.y = s.sy = 0;
    }
    else
    {
        int ty = r.y - 2;
        s.y = ty >> 2;
        s.sy = ((ty + r.sy) >> 2) - s.y + 2;
    }
    return s;
}

void initBlendedNoise(BlendedNoise *bn, uint64_t ws, int dim)
{
    static const int jhash_terrain = 0x62d03a68; // minecraft:terrain
    static const uint64_t md5_terrain[2] = {0x1ee555222ef96f14, 0xe2bedfdbebe43d33}; // minecraft:terrain

    switch (dim) {
    case DIM_OVERWORLD:
        bn->xzScale = 0.25;
        bn->yScale = 0.125;
        bn->xzFactor = 80.0;
        bn->yFactor = 160.0;
        bn->smearScaleMultiplier = 8.0;
        break;
    case DIM_NETHER:
        bn->xzScale = 0.25;
        bn->yScale = 0.375;
        bn->xzFactor = 80.0;
        bn->yFactor = 60.0;
        bn->smearScaleMultiplier = 8.0;
        break;
    case DIM_END:
        bn->xzScale = 0.25;
        bn->yScale = 0.25;
        bn->xzFactor = 80.0;
        bn->yFactor = 160.0;
        bn->smearScaleMultiplier = 4.0;
        break;
    default: UNREACHABLE();
    }
    bn->xzMultiplier = 684.412 * bn->xzScale;
    bn->yMultiplier = 684.412 * bn->yScale;
    bn->smearedYScale = bn->yMultiplier * bn->smearScaleMultiplier;
    bn->factoredSmearedYScale = bn->smearedYScale / bn->yFactor;

    switch (dim) {
    case DIM_OVERWORLD: {
        Xoroshiro wsx;
        xSetSeed(&wsx, ws);
        const uint64_t lo = xNextLong(&wsx);
        const uint64_t hi = xNextLong(&wsx);

        Xoroshiro xr = {lo ^ md5_terrain[0], hi ^ md5_terrain[1]};
        xOctaveLegacyInit(&bn->octmin, &xr, bn->oct+0, -15, 16);
        xOctaveLegacyInit(&bn->octmax, &xr, bn->oct+16, -15, 16);
        xOctaveLegacyInit(&bn->octmain, &xr, bn->oct+32, -7, 8);
        break;
    }
    case DIM_NETHER: {
        uint64_t wsj;
        setSeed(&wsj, ws);
        octaveInit(&bn->octmin, &wsj, bn->oct+0, -15, 16);
        octaveInit(&bn->octmax, &wsj, bn->oct+16, -15, 16);
        octaveInit(&bn->octmain, &wsj, bn->oct+32, -7, 8);
        break;
    }
    case DIM_END: {
        uint64_t wsr;
        setSeed(&wsr, ws);
        const uint64_t nl = nextLong(&wsr);

        uint64_t rnd;
        setSeed(&rnd, jhash_terrain ^ nl);
        octaveInit(&bn->octmin, &rnd, bn->oct+0, -15, 16);
        octaveInit(&bn->octmax, &rnd, bn->oct+16, -15, 16);
        octaveInit(&bn->octmain, &rnd, bn->oct+32, -7, 8);
        break;
    }
    default: UNREACHABLE();
    }
}

double sampleBase3dNoise(BlendedNoise *bn, int x, int y, int z)
{
    const double scaledX = x * bn->xzMultiplier;
    const double scaledY = y * bn->yMultiplier;
    const double scaledZ = z * bn->xzMultiplier;
    const double factoredX = scaledX / bn->xzFactor;
    const double factoredY = scaledY / bn->yFactor;
    const double factoredZ = scaledZ / bn->xzFactor;

    double minSampleTotal = 0.0;
    double maxSampleTotal = 0.0;
    double mainSampleTotal = 0.0;
    double o = 1.0;

    for (int octaveIdx = 0; octaveIdx < 8; octaveIdx++) {
        mainSampleTotal += samplePerlin(&bn->octmain.octaves[octaveIdx], maintainPrecision(factoredX * o), maintainPrecision(factoredY * o), maintainPrecision(factoredZ * o), bn->factoredSmearedYScale * o, factoredY * o) / o;
        o /= 2.0;
    }

    double q = (mainSampleTotal / 10.0 + 1.0) / 2.0;
    o = 1.0;

    for (int octaveIdx = 0; octaveIdx < 16; octaveIdx++) {
        double sampleY = maintainPrecision(scaledY * o);
        double sampleX = maintainPrecision(scaledX * o);
        double sampleZ = maintainPrecision(scaledZ * o);
        double yamp = bn->smearedYScale * o;
        double ymax = scaledY * o;
        if (q < 1.0) {
            minSampleTotal += samplePerlin(&bn->octmin.octaves[octaveIdx], sampleX, sampleY, sampleZ, yamp, ymax) / o;
        }

        if (q > 0.0) {
            maxSampleTotal += samplePerlin(&bn->octmax.octaves[octaveIdx], sampleX, sampleY, sampleZ, yamp, ymax) / o;
        }

        o /= 2.0;
    }

    return clampedLerp(q, minSampleTotal / 512.0, maxSampleTotal / 512.0) / 128.0;
}
