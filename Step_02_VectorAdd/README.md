# Step 02: Vector Addition

In this step, you will implement element-wise addition of two vectors ($C = A + B$) on the GPU. This is the classic parallel processing introductory task.

## Concepts to Learn

### 1. GPU Memory Allocation (`cudaMalloc`)
The GPU (device) has its own physical DRAM separate from the CPU (host) RAM. To store data on the GPU, you must allocate memory using:
```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size);
```
* `devPtr`: Address of a pointer that will hold the allocated device address.
* `size`: Number of bytes to allocate.

### 2. Host-Device Data Transfer (`cudaMemcpy`)
To move data between host and device, use:
```cpp
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```
* `kind` specifies the direction of copy:
  * `cudaMemcpyHostToDevice` (CPU → GPU)
  * `cudaMemcpyDeviceToHost` (GPU → CPU)

### 3. GPU Memory Deallocation (`cudaFree`)
Always free allocated GPU memory to prevent memory leaks:
```cpp
cudaError_t cudaFree(void* devPtr);
```

### 4. Global Thread Indexing in 1D Grid
When executing in parallel, each thread must know which element of the vector it is responsible for. To calculate a unique 1D index:
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

### 5. Grid/Block Size Calculation
If your vector size $N$ is not a multiple of the block size (e.g., $N = 50000$, and block size is $256$), you need to calculate the number of blocks dynamically.
To ensure you cover all elements without off-by-one errors or truncation, use integer ceiling division:
$$\text{gridDim} = \frac{N + \text{blockDim} - 1}{\text{blockDim}}$$
In C++:
```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
```
*Since this might launch more threads than $N$ (e.g., $196 \times 256 = 50176$ threads), you **must** guard against out-of-bounds memory access inside your kernel:*
```cpp
if (i < N) { ... }
```

---

## Code Skeletons & Pseudocode

### 1. `src/kernel.h`
Declare the wrapper function that `main.cpp` will call to interface with the CUDA kernel:
* Parameters: Host input pointers ($A$, $B$), host output pointer ($C$), and vector size ($N$).

### 2. `src/kernel.cu`

#### A. The CUDA Kernel
```cpp
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    // 1. Calculate global 1D thread index
    // 2. If index < N, perform addition: C[index] = A[index] + B[index]
}
```

#### B. The C++ Launcher Wrapper
```cpp
void launchVectorAdd(const float* h_A, const float* h_B, float* h_C, int N) {
    // 1. Declare device pointers (d_A, d_B, d_C)
    // 2. Allocate device memory using cudaMalloc
    // 3. Copy h_A and h_B to d_A and d_B using cudaMemcpyHostToDevice
    // 4. Set threads_per_block = 256, calculate blocks_per_grid
    // 5. Launch vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N)
    // 6. Check for errors and synchronize with cudaDeviceSynchronize()
    // 7. Copy device result d_C back to h_C using cudaMemcpyDeviceToHost
    // 8. Free device memory (d_A, d_B, d_C) using cudaFree
}
```

### 3. `src/main.cpp`
```cpp
int main() {
    // 1. Set size N (e.g., 50000)
    // 2. Allocate host memory (std::vector<float> or raw pointers)
    // 3. Initialize host vectors with test numbers
    // 4. Call launchVectorAdd()
    // 5. Verify the GPU result on the CPU:
    //    Loop i from 0 to N: check if h_C[i] == h_A[i] + h_B[i]
    // 6. Print success or failure
}
```

---

## Task List for User

- [ ] Complete `src/kernel.h` declaring the launcher wrapper.
- [ ] Implement the kernel `vectorAddKernel` and the wrapper function in `src/kernel.cu`.
- [ ] Set up the memory allocations, launch execution, and results verification in `src/main.cpp`.
- [ ] Run `build.bat` inside the `Step_02_VectorAdd` folder.
- [ ] Execute `build\vector_add.exe` and check if verification succeeds.
