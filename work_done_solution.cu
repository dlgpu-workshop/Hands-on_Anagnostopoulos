#include <stdio.h>
#include <cublas_v2.h>

#define N 1

int main() {
    float h_F[N] = {5.0f};  // Host array for force
    float h_d[N] = {10.0f};  // Host array for displacement
    float *d_F, *d_d;  // Device arrays
    float result;  // Result of the dot product
    cublasHandle_t handle;  // cuBLAS handle

    // Allocate device memory
    cudaMalloc((void**)&d_F, N * sizeof(float));
    cudaMalloc((void**)&d_d, N * sizeof(float));

    // Copy vectors to device
    cudaMemcpy(d_F, h_F, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasCreate(&handle);

    // Calculate dot product
    cublasSdot(handle, N, d_F, 1, d_d, 1, &result);

    // Print result
    printf("Work done: %f J\n", result);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_F);
    cudaFree(d_d);

    return 0;
}

