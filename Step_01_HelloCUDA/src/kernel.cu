#include <cstdio>

#include <cuda_runtime.h>

#include "kernel.h"

__global__ void helloFromGPU()
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Hello from GPU! threadId: %d , blockId: %d , blockDim: %d , gridDim: %d , threadIdx: %d\n", threadId, blockIdx.x, blockDim.x, gridDim.x, threadIdx.x);
}

// function launchHelloKernel should get the device id
// and call the kernel on that device
void launchHelloKernel()
{
    int num_blocks = 4;
    int threads_per_block = 8; // better to be power of 2

    helloFromGPU<<<num_blocks, threads_per_block>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error after launch: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA Error after synchronize: %s\n", cudaGetErrorString(err));
    }
}
