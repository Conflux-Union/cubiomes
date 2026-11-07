/*
 * Runtime-bound OpenCL implementation, see opencl_host.h.
 */

#include "opencl_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void *gpuClOpenLib(void)
{
#if defined(_WIN32)
    return (void *) LoadLibraryA("OpenCL.dll");
#else
    void *lib;
#if defined(__APPLE__)
    /* The framework path exists on every macOS release with OpenCL. */
    lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL",
        RTLD_LAZY | RTLD_LOCAL);
    if (lib)
        return lib;
    return dlopen("libOpenCL.dylib", RTLD_LAZY | RTLD_LOCAL);
#else
    lib = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (lib)
        return lib;
    return dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);
#endif
#endif
}

static void *gpuClSym(void *lib, const char *name)
{
#if defined(_WIN32)
    return (void *) GetProcAddress((HMODULE) lib, name);
#else
    return dlsym(lib, name);
#endif
}

#define GPUCL_SYM(fname) \
    do { cl->fn.fname = (cl##fname##_t) (intptr_t) gpuClSym(lib, "cl" #fname); \
         if (!cl->fn.fname) { fprintf(stderr, "gpu: missing symbol cl%s\n", #fname); \
             return 1; } } while (0)

int gpuClInit(GpuCl *cl)
{
    memset(cl, 0, sizeof(*cl));

    void *lib = gpuClOpenLib();
    if (!lib)
        return 1;

    cl->fn.lib = lib;
    GPUCL_SYM(GetPlatformIDs);
    GPUCL_SYM(GetDeviceIDs);
    GPUCL_SYM(GetDeviceInfo);
    GPUCL_SYM(CreateContext);
    GPUCL_SYM(ReleaseContext);
    GPUCL_SYM(CreateCommandQueue);
    GPUCL_SYM(ReleaseCommandQueue);
    GPUCL_SYM(CreateBuffer);
    GPUCL_SYM(ReleaseMemObject);
    GPUCL_SYM(CreateProgramWithSource);
    GPUCL_SYM(BuildProgram);
    GPUCL_SYM(GetProgramBuildInfo);
    GPUCL_SYM(ReleaseProgram);
    GPUCL_SYM(CreateKernel);
    GPUCL_SYM(ReleaseKernel);
    GPUCL_SYM(SetKernelArg);
    GPUCL_SYM(EnqueueNDRangeKernel);
    GPUCL_SYM(EnqueueReadBuffer);
    GPUCL_SYM(GetKernelWorkGroupInfo);

    cl_platform_id plats[8];
    cl_uint nplats = 0;
    if (cl->fn.GetPlatformIDs(8, plats, &nplats) != CL_SUCCESS || nplats == 0)
        return 1;

    int found = 0;
    for (cl_uint p = 0; p < nplats && !found; p++)
    {
        cl_device_id devs[8];
        cl_uint ndevs = 0;
        if (cl->fn.GetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 8, devs,
                &ndevs) != CL_SUCCESS)
            continue;
        for (cl_uint d = 0; d < ndevs; d++)
        {
            cl_ulong fpconf = 0;
            cl->fn.GetDeviceInfo(devs[d], CL_DEVICE_DOUBLE_FP_CONFIG,
                sizeof(fpconf), &fpconf, NULL);
            if (!fpconf)
                continue; /* no fp64: the engine is a pure executor or nothing */
            cl->platform = plats[p];
            cl->device = devs[d];
            cl->fp64 = 1;
            found = 1;
            break;
        }
    }
    if (!found)
        return 1;

    cl->fn.GetDeviceInfo(cl->device, CL_DEVICE_NAME,
        sizeof(cl->devName) - 1, cl->devName, NULL);
    cl->fn.GetDeviceInfo(cl->device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cl->cus), &cl->cus, NULL);

    cl_int err;
    cl->context = cl->fn.CreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
    if (!cl->context || err != CL_SUCCESS)
        return 1;
    cl->queue = cl->fn.CreateCommandQueue(cl->context, cl->device, 0, &err);
    if (!cl->queue || err != CL_SUCCESS)
    {
        cl->fn.ReleaseContext(cl->context);
        cl->context = NULL;
        return 1;
    }
    return 0;
}

int gpuClBuild(GpuCl *cl, const char *src, const char *opts,
    const char *kernelName)
{
    if (cl->kernel)
    {
        cl->fn.ReleaseKernel(cl->kernel);
        cl->kernel = NULL;
    }
    if (cl->program)
    {
        cl->fn.ReleaseProgram(cl->program);
        cl->program = NULL;
    }

    size_t len = strlen(src);
    cl_int err;
    cl->program = cl->fn.CreateProgramWithSource(cl->context, 1, &src, &len,
        &err);
    if (!cl->program || err != CL_SUCCESS)
        return 1;

    err = cl->fn.BuildProgram(cl->program, 1, &cl->device, opts, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t logLen = 0;
        cl->fn.GetProgramBuildInfo(cl->program, cl->device,
            CL_PROGRAM_BUILD_LOG, 0, NULL, &logLen);
        if (logLen > 0)
        {
            char *log = (char *) malloc(logLen + 1);
            if (log)
            {
                cl->fn.GetProgramBuildInfo(cl->program, cl->device,
                    CL_PROGRAM_BUILD_LOG, logLen, log, NULL);
                log[logLen] = 0;
                fprintf(stderr, "gpu: kernel build failed:\n%s\n", log);
                free(log);
            }
        }
        cl->fn.ReleaseProgram(cl->program);
        cl->program = NULL;
        return 1;
    }

    cl->kernel = cl->fn.CreateKernel(cl->program, kernelName, &err);
    if (!cl->kernel || err != CL_SUCCESS)
    {
        cl->fn.ReleaseProgram(cl->program);
        cl->program = NULL;
        return 1;
    }
    return 0;
}
