# Step 01: Hello CUDA (with main.cpp entry point)

Welcome to the first step of your CUDA journey! In this step, we will verify your CUDA toolchain and learn how to separate CPU host code (`main.cpp`) from GPU device code (`kernel.cu`).

## Concepts to Learn

### 1. Separation of Host and Device Code
To keep your project structured like a professional C++ application, we keep the main entry point and CPU-only code in `.cpp` files, while the GPU kernels and wrapper launch functions reside in `.cu` files.

Since standard C++ compilers (like MSVC `cl.exe`) don't understand CUDA kernel syntax (`<<<...>>>` and `__global__`), we cannot launch kernels directly from `.cpp` files. Instead, we:
1. Define the kernel in a `.cu` file.
2. Define a standard C++ wrapper function in the same `.cu` file that launches the kernel.
3. Declare this wrapper function in a header file (or directly in `main.cpp` as an `extern` function).
4. Call the wrapper function from `main.cpp`.

---

## Folder Structure for Step 01

```
Step_01_HelloCUDA/
├── src/
│   ├── main.cpp     ← C++ Entry point, queries device props, calls wrapper
│   ├── kernel.cu    ← CUDA kernel and the wrapper function implementation
│   └── kernel.h     ← Declares the wrapper function
└── build.bat
```

---

## Your Task

I've created the three source files for you. Let's write the code!

### 1. `src/kernel.h`
Declare the C++ wrapper function here. It should be a standard C++ function:
```cpp
#pragma once

void launchHelloKernel();
```

### 2. `src/kernel.cu`
Implement the CUDA kernel and the wrapper function here.
- Include `<cstdio>` (for `printf`) and `"kernel.h"`.
- Write the `__global__` kernel:
  ```cpp
  __global__ void helloFromGPU() {
      printf("Hello from GPU thread %d\n", threadIdx.x);
  }
  ```
- Implement `launchHelloKernel()`:
  - Launch `helloFromGPU<<<1, 10>>>();`.
  - Synchronize the device: `cudaDeviceSynchronize();`.

### 3. `src/main.cpp`
Write the CPU entry point.
- Include `<iostream>`, `<cuda_runtime.h>` (for device properties), and `"kernel.h"`.
- In `main()`:
  - Print `"Hello from CPU!"`.
  - Call `launchHelloKernel()`.
  - Query device properties using `cudaGetDeviceProperties` and print the GPU name.
  - Return 0.
