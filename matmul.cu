#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>

#define N 1500
#define BLOCK_SIZE 16

// CPU function
void matmul_cpu(float *A, float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA kernel
__global__ void matmul_gpu(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

int main() {
    float *A, *B, *C, *D, *E;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    clock_t start, end;

    // Allocate memory
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));
    cudaMalloc((void**)&D, N * N * sizeof(float));
    cudaMalloc((void**)&E, N * N * sizeof(float));

    // Initialize A and B... using values between 1 and 10:
    srand(time(0));  // seed random number generator
    for (int i = 0; i < N * N; i++) {
        A[i] = 1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10-1)));
        B[i] = 1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10-1)));
    }

    // CPU execution
    start = clock();
    matmul_cpu(A, B, C);
    end = clock();
    printf("CPU execution time:\t %f seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // CUDA execution
    cudaMemcpy(D, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(E, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / threads.x, N / threads.y);
    start = clock();
    matmul_gpu<<<grid, threads>>>(D, E, D);
    cudaDeviceSynchronize();
    end = clock();
    printf("CUDA execution time:\t %f seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // cuBLAS execution
    cublasCreate(&handle);
    start = clock();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, D, N, E, N, &beta, D, N);
    cudaDeviceSynchronize();
    end = clock();
    printf("cuBLAS execution time:\t %f seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    cublasDestroy(handle);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(D);
    cudaFree(E);

    return 0;
}
