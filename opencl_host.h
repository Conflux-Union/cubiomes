#ifndef OPENCL_HOST_H_
#define OPENCL_HOST_H_

/*
 * Minimal runtime-bound OpenCL layer for the GPU engine.
 *
 * Resolves the platform's OpenCL loader via dlopen/LoadLibrary at startup:
 * libOpenCL.so.1 (+ plain .so) on Linux, the OpenCL framework on macOS,
 * OpenCL.dll on Windows. Declares only the types, constants and entry points
 * the engine needs (~40 lines of own declarations) - Khronos headers are not
 * vendored and nothing is linked against OpenCL. Every failure (no loader, no
 * GPU device, no fp64) surfaces as a nonzero return and the caller falls back
 * to the CPU path.
 */

#include <stddef.h>
#include <stdint.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint16_t cl_bool;

typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

typedef intptr_t cl_context_properties;

enum
{
    CL_SUCCESS = 0,
    CL_TRUE = 1,
    CL_FALSE = 0,

    CL_DEVICE_TYPE_GPU = (1 << 2),
    CL_DEVICE_NAME = 0x102B,
    CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002,
    CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032,

    CL_MEM_READ_WRITE = (1 << 0),
    CL_MEM_READ_ONLY = (1 << 2),
    CL_MEM_COPY_HOST_PTR = (1 << 5),

    CL_PROGRAM_BUILD_LOG = 0x1183,
    CL_KERNEL_WORK_GROUP_SIZE = 0x11B0,
};

typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_ulong, cl_uint,
    cl_device_id *, cl_uint *);
typedef cl_int (*clGetDeviceInfo_t)(cl_device_id, cl_uint, size_t, void *,
    size_t *);
typedef cl_context (*clCreateContext_t)(const cl_context_properties *, cl_uint,
    const cl_device_id *, void *, void *, cl_int *);
typedef cl_int (*clReleaseContext_t)(cl_context);
typedef cl_command_queue (*clCreateCommandQueue_t)(cl_context, cl_device_id,
    cl_ulong, cl_int *);
typedef cl_int (*clReleaseCommandQueue_t)(cl_command_queue);
typedef cl_mem (*clCreateBuffer_t)(cl_context, cl_ulong, size_t, void *,
    cl_int *);
typedef cl_int (*clReleaseMemObject_t)(cl_mem);
typedef cl_program (*clCreateProgramWithSource_t)(cl_context, cl_uint,
    const char **, const size_t *, cl_int *);
typedef cl_int (*clBuildProgram_t)(cl_program, cl_uint, const cl_device_id *,
    const char *, void *, void *);
typedef cl_int (*clGetProgramBuildInfo_t)(cl_program, cl_device_id, cl_uint,
    size_t, void *, size_t *);
typedef cl_int (*clReleaseProgram_t)(cl_program);
typedef cl_kernel (*clCreateKernel_t)(cl_program, const char *, cl_int *);
typedef cl_int (*clReleaseKernel_t)(cl_kernel);
typedef cl_int (*clSetKernelArg_t)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel,
    cl_uint, const size_t *, const size_t *, const size_t *, cl_uint,
    const cl_event *, cl_event *);
typedef cl_int (*clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_bool,
    size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*clGetKernelWorkGroupInfo_t)(cl_kernel, cl_device_id,
    cl_uint, size_t, void *, size_t *);

typedef struct
{
    void *lib;

    clGetPlatformIDs_t GetPlatformIDs;
    clGetDeviceIDs_t GetDeviceIDs;
    clGetDeviceInfo_t GetDeviceInfo;
    clCreateContext_t CreateContext;
    clReleaseContext_t ReleaseContext;
    clCreateCommandQueue_t CreateCommandQueue;
    clReleaseCommandQueue_t ReleaseCommandQueue;
    clCreateBuffer_t CreateBuffer;
    clReleaseMemObject_t ReleaseMemObject;
    clCreateProgramWithSource_t CreateProgramWithSource;
    clBuildProgram_t BuildProgram;
    clGetProgramBuildInfo_t GetProgramBuildInfo;
    clReleaseProgram_t ReleaseProgram;
    clCreateKernel_t CreateKernel;
    clReleaseKernel_t ReleaseKernel;
    clSetKernelArg_t SetKernelArg;
    clEnqueueNDRangeKernel_t EnqueueNDRangeKernel;
    clEnqueueReadBuffer_t EnqueueReadBuffer;
    clGetKernelWorkGroupInfo_t GetKernelWorkGroupInfo;
} GpuClFuncs;

/*
 * Live OpenCL runtime state: one device, one context, one command queue.
 * The built program/kernel is cached here; gpuClBuild() (re)builds it.
 */
typedef struct
{
    GpuClFuncs fn;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    char devName[256];
    cl_uint cus;
    int fp64;
} GpuCl;

/* Resolves the OpenCL loader, probes the first GPU device (fp64 flag, name,
 * compute units) and creates a context plus one command queue.
 * Returns 0 on success, nonzero when OpenCL/a usable GPU is not available. */
int gpuClInit(GpuCl *cl);

/* Builds `src` with `opts` on the probed device and caches program + kernel.
 * A build failure is reported with the compiler log on stderr.
 * Returns 0 on success. */
int gpuClBuild(GpuCl *cl, const char *src, const char *opts,
    const char *kernelName);

#endif
