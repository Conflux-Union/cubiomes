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
    GPUCL_SYM(EnqueueWriteBuffer);
    GPUCL_SYM(WaitForEvents);
    GPUCL_SYM(ReleaseEvent);
    GPUCL_SYM(GetEventProfilingInfo);
    cl->fn.EnqueueMarkerWithWaitList = (clEnqueueMarkerWithWaitList_t)
        (intptr_t) gpuClSym(lib, "clEnqueueMarkerWithWaitList");
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
        /* First GPU device wins: float-only kernels make every OpenCL GPU
         * eligible. */
        cl->platform = plats[p];
        cl->device = devs[0];
        found = 1;
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
    /* Event-profiling opt-in for the metrics timers (gpu.h slots 9-12):
     * CUBIOMES_GPU_PROFILING=1 creates the queue with
     * CL_QUEUE_PROFILING_ENABLE so clGetEventProfilingInfo reports device
     * timestamps. Anything else keeps the plain queue. */
    const char *profEnv = getenv("CUBIOMES_GPU_PROFILING");
    cl->profiling = profEnv && !strcmp(profEnv, "1");
    cl->queue = cl->fn.CreateCommandQueue(cl->context, cl->device,
        cl->profiling ? (cl_ulong) CL_QUEUE_PROFILING_ENABLE : 0, &err);
    if (!cl->queue || err != CL_SUCCESS)
    {
        cl->fn.ReleaseContext(cl->context);
        cl->context = NULL;
        return 1;
    }
    /* Some loader/driver stacks accept the profiling flag yet still answer
     * every clGetEventProfilingInfo with CL_PROFILING_INFO_NOT_AVAILABLE
     * (observed: NVIDIA 580.119.02's own libOpenCL loader on Deepin, where
     * the flag even reads back set through clCreateCommandQueue). Probe the
     * actual capability with a marker command and degrade to unprofiled
     * when the queue cannot deliver device timestamps. */
    if (cl->profiling)
    {
        cl_event probe = NULL;
        cl_int perr = -1;  /* sentinel: no marker entry in this loader */
        if (cl->fn.EnqueueMarkerWithWaitList)
            perr = cl->fn.EnqueueMarkerWithWaitList(cl->queue, 0, NULL,
                &probe);
        if (perr == CL_SUCCESS && probe)
        {
            cl_ulong end = 0;
            if (cl->fn.WaitForEvents(1, &probe) == CL_SUCCESS)
                cl->profiling = cl->fn.GetEventProfilingInfo(probe,
                    CL_PROFILING_COMMAND_END, sizeof(end), &end,
                    NULL) == CL_SUCCESS;
            else
                cl->profiling = 0;
            cl->fn.ReleaseEvent(probe);
        }
        else
            cl->profiling = 0;
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
