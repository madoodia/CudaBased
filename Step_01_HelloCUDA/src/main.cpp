#include <iostream>

#include <cuda_runtime.h>

#include "kernel.h"

int main()
{
    // getting all devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of devices: " << deviceCount << std::endl;
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }

    cudaSetDevice(0); // Select device 0
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;

    launchHelloKernel(); // Launch kernel

    return 0;
}