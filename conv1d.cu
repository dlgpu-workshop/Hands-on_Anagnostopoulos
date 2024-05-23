#include <stdio.h>
#include <cudnn.h>
#include <time.h>

#define N 10000000
#define M 1000
#define BLOCK_SIZE 256

// CUDA kernel for 1D convolution
__global__ void conv1d_gpu(float *X, float *F, float *Y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0;
        for (int j = 0; j < M; j++) {
            if (i - j >= 0) {
                sum += X[i - j] * F[j];
            }
        }
        Y[i] = sum;
    }
}

// CPU function for 1D convolution
void conv1d_cpu(float *X, float *F, float *Y) {
    for (int i = 0; i < N; i++) {
        float sum = 0;
        for (int j = 0; j < M; j++) {
            if (i - j >= 0) {
                sum += X[i - j] * F[j];
            }
        }
        Y[i] = sum;
    }
}



int main() {
    float *h_X, *h_F, *h_Y, *d_X, *d_F, *d_Y;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t fDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    
    size_t workspaceSize;
    void *workspace;
    clock_t start, end;
    double cpu_time_used, cuda_time_used, cudnn_time_used;

    // Allocate host and device memory
    h_X = (float*)malloc(N * sizeof(float));
    h_F = (float*)malloc(M * sizeof(float));
    h_Y = (float*)malloc(N * sizeof(float));
    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_F, M * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    // Initialize h_X and h_F...
    for (int i = 0; i < N; i++) {
        h_X[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M; i++) {
        h_F[i] = (float)rand() / RAND_MAX;
    }


    // Copy to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, M * sizeof(float), cudaMemcpyHostToDevice);

    // Perform 1D convolution using CUDA
    start = clock();
    conv1d_gpu<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_X, d_F, d_Y);
    cudaDeviceSynchronize();
    end = clock();
    cuda_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CUDA Time:\t %f\n", cuda_time_used);

    // Copy back to host
    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform 1D convolution using cuDNN
    // Create cuDNN handle and descriptors
    cudnnCreate(&handle);
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateFilterDescriptor(&fDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    // Set tensor and filter descriptors
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N);
    cudnnSetFilter4dDescriptor(fDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, M);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // Choose convolution algorithm for version before cudnn 8
    // cudnnGetConvolutionForwardAlgorithm(handle, xDesc, fDesc, convDesc, yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    

    // Allocate workspace
    cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, fDesc, convDesc, yDesc, algo, &workspaceSize);
    cudaMalloc(&workspace, workspaceSize);

    // Perform 1D convolution using cuDNN
    start = clock();
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, xDesc, d_X, fDesc, d_F, convDesc, algo, workspace, workspaceSize, &beta, yDesc, d_Y);
    cudaDeviceSynchronize();
    end = clock();
    cudnn_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("cuDNN Time:\t %f\n", cudnn_time_used);

    // Copy back to host
    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform 1D convolution on CPU
    start = clock();
    conv1d_cpu(h_X, h_F, h_Y);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time:\t %f\n", cpu_time_used);

    // Cleanup
    cudaFree(workspace);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(fDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle);
    free(h_X);
    free(h_F);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_F);
    cudaFree(d_Y);

    return 0;
}
