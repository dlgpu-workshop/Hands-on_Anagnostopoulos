/*
This program first allocates memory for a large array
of integers on both the CPU and GPU. It then initializes
the array with some values. The square_cpu function squares
each element of the array on the CPU, and the square_gpu
kernel does the same on the GPU. The program measures and prints
the execution times for both the CPU and GPU computations

Includes error checking
*/

#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define N 100000000
#define BLOCK_SIZE 1024

// Error checking macro
#define CUDA_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CPU function
void square_cpu(int *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = a[i] * a[i];
    }
}

// GPU kernel
__global__ void square_gpu(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] * a[idx];
    }
}

int main() {
    int *a_cpu, *a_gpu;
    int size = N * sizeof(int);

    // Allocate memory
    a_cpu = (int*)malloc(size);
    CUDA_ERROR(cudaMalloc((void**)&a_gpu, size));

    // Initialize array
    for (int i = 0; i < N; i++) {
        a_cpu[i] = i;
    }
    CUDA_ERROR(cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice));

    // Print CUDA version
    printf("CUDA version: %d.%d\n", CUDA_VERSION / 1000, (CUDA_VERSION % 100) / 10);

    // CPU execution
    clock_t start_cpu = clock();
    square_cpu(a_cpu, N);
    clock_t end_cpu = clock();

    // GPU execution
    clock_t start_gpu = clock();
    square_gpu<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(a_gpu, N);
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaDeviceSynchronize());
    clock_t end_gpu = clock();

    // Print execution times
    double time_cpu = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    double time_gpu = ((double) (end_gpu - start_gpu)) / CLOCKS_PER_SEC;
    printf("CPU execution time: %f seconds\n", time_cpu);
    printf("GPU execution time: %f seconds\n", time_gpu);

    // Free memory
    free(a_cpu);
    CUDA_ERROR(cudaFree(a_gpu));

    return 0;
}
