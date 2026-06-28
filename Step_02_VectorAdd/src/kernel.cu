#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include "kernel.h"

// ============================================================================
// 1. THE GPU KERNEL (Runs on the GPU hardware)
// ============================================================================
// - __global__ tells the compiler this function is called from the CPU (Host)
//   but runs on the GPU (Device).
// - It must return 'void'.
// - Since it runs on the GPU, it cannot access Host pointers (h_A, h_B, h_C).
//   It must only operate on Device pointers (d_A, d_B, d_C).
// ============================================================================

__global__ void vectorAddKernel(const float* d_A, const float* d_B, float* d_C, int N)
{
    // a. We need to find which unique element of the array this thread is responsible for.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // b. Because we launch blocks of size 256, the total threads launched might exceed N.
    if (idx < N)
    {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

// ============================================================================
// 2. THE HOST WRAPPER (Runs on CPU, orchestrates the GPU execution)
// ============================================================================
// Add this helper struct at the top of kernel.cu
struct DeviceBuffer
{
    float* ptr = nullptr;

    // Destructor automatically frees the GPU memory when the object is destroyed
    ~DeviceBuffer()
    {
        if (ptr)
            cudaFree(ptr);
    }
};

void launchVectorAdd(const float* h_A, const float* h_B, float* h_C, int N)
{
    size_t size = N * sizeof(float);

    // Objects are allocated on the CPU stack
    DeviceBuffer d_A, d_B, d_C;

    // Allocate GPU memory into the objects
    if (cudaMalloc(&d_A.ptr, size) != cudaSuccess)
        return;
    if (cudaMalloc(&d_B.ptr, size) != cudaSuccess)
        return;
    if (cudaMalloc(&d_C.ptr, size) != cudaSuccess)
        return;

    // Copy data
    if (cudaMemcpy(d_A.ptr, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
        return;
    if (cudaMemcpy(d_B.ptr, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
        return;

    // Launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A.ptr, d_B.ptr, d_C.ptr, N);

    if (cudaDeviceSynchronize() != cudaSuccess)
        return;

    // Copy back
    if (cudaMemcpy(h_C, d_C.ptr, size, cudaMemcpyDeviceToHost) != cudaSuccess)
        return;

    // No need to call cudaFree manually!
    // When the function returns (either here or early), d_A, d_B, and d_C destructors run automatically.
}