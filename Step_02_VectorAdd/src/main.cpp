#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "kernel.h"

// ============================================================================
// MAIN CPU ENTRY POINT
// ============================================================================
int main()
{
    int N = 10000000; // 1 Million elements (large enough to show GPU speedup)

    std::cout << "Vector addition of size: " << N << std::endl;

    // a. Declare the host arrays to hold our data.
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N, 0.0f); // Initialize C with zeros

    // TODO: Review and implement SOA Methodology
    // struct Vector3
    // {
    //     std::vector<float> x;
    //     std::vector<float> y;
    //     std::vector<float> z;
    // };
    // Vector3 h_A_soa;
    // Vector3 h_B_soa;
    // Vector3 h_C_soa;
    // h_A_soa.x.resize(N);
    // h_A_soa.y.resize(N);
    // h_A_soa.z.resize(N);
    // h_B_soa.x.resize(N);
    // h_B_soa.y.resize(N);
    // h_B_soa.z.resize(N);
    // h_C_soa.x.resize(N);
    // h_C_soa.y.resize(N);
    // h_C_soa.z.resize(N);

    // b. Initialize the host arrays with dummy values.
    for (int i = 0; i < N; i++)
    {
        h_A[i] = sinf(i) * 2.0f;
        h_B[i] = cosf(i) * 3.0f;
    }

    // c. Call the host wrapper function we declared in kernel.h.
    launchVectorAdd(h_A.data(), h_B.data(), h_C.data(), N);

    // d. Verify that the GPU results are mathematically correct.
    int errors = 0;
    float epsilon = 1e-5f;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > epsilon)
        {
            errors++;
        }
    }
    if (errors == 0)
    {
        std::cout << "Verification Succeeded!" << std::endl;
    } else
    {
        std::cout << "Verification Failed!" << std::endl;
    }

    return 0;
}
