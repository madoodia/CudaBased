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
void launchVectorAdd(const float* h_A, const float* h_B, float* h_C, int N)
{
    size_t size = N * sizeof(float);

    // a. Declare the device pointers that will point to memory allocated in VRAM.
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    // b. Allocate memory on the GPU for all three arrays.
    if (cudaMalloc(&d_A, size) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory on the GPU." << std::endl;
        return;
    }
    if (cudaMalloc(&d_B, size) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory on the GPU." << std::endl;
        return;
    }
    if (cudaMalloc(&d_C, size) != cudaSuccess)
    {
        std::cerr << "Failed to allocate memory on the GPU." << std::endl;
        return;
    }

    // c. Copy input vectors A and B from CPU RAM (Host) to GPU VRAM (Device).
    if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from CPU to GPU." << std::endl;
        return;
    }
    if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from CPU to GPU." << std::endl;
        return;
    }

    // d. Configure the execution grid layout.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // e. Launch the GPU kernel.
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // f. Wait for the GPU to finish execution and check for errors.
    cudaError_t syncErr = cudaDeviceSynchronize();
    cudaError_t asyncErr = cudaGetLastError();
    if (syncErr != cudaSuccess)
    {
        std::cerr << "Device synchronization failed: " << cudaGetErrorString(syncErr) << std::endl;
    }
    if (asyncErr != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(asyncErr) << std::endl;
    }

    // g. Copy the output vector C from GPU VRAM (Device) back to CPU RAM (Host).
    if (cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Failed to copy data from GPU to CPU." << std::endl;
        return;
    }

    // h. Free the allocated memory on the GPU to prevent VRAM memory leaks.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
