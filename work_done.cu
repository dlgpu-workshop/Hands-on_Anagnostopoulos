#include <stdio.h>
#include <cublas_v2.h>

#define N 1

int main() {
    float h_F[N] = {5.0f};  // Host array for force
    float h_d[N] = {10.0f};  // Host array for displacement
    float *d_F, *d_d;  // Device arrays
    float result;  // Result of the dot product
    cublasHandle_t handle;  // cuBLAS handle

    // Allocate device memory with cudaMalloc for d_F and d_d


    // Copy vectors to device with cudaMemcpy


    // Create cuBLAS handle


    // Calculate dot product using cublasSdot


    // Print result
    printf("Work done: %f J\n", result);


    // Cleanup hanlde nad device memory



    return 0;
}

