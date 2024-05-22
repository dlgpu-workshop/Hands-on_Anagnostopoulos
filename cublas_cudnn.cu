#include <cublas_v2.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 3
#define K 2
#define N 4

void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Function to initialize a matrix with random values between -1 and 1
void initializeMatrix(float* matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / (float)(RAND_MAX / 2) - 1.0f;
    }
}

int main() {
    // Initialize cuBLAS and cuDNN
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);

    // Allocate and initialize input matrices on the device
    int rowsA = M, colsA = K, rowsB = K, colsB = N;
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, sizeof(float) * rowsA * colsA);
    cudaMalloc(&d_B, sizeof(float) * rowsB * colsB);
    cudaMalloc(&d_C, sizeof(float) * rowsA * colsB);

    // Initialize d_A and d_B with random values between -1 and 1
    float* h_A = (float*)malloc(sizeof(float) * rowsA * colsA);
    float* h_B = (float*)malloc(sizeof(float) * rowsB * colsB);
    initializeMatrix(h_A, rowsA, colsA);
    initializeMatrix(h_B, rowsB, colsB);
    cudaMemcpy(d_A, h_A, sizeof(float) * rowsA * colsA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * rowsB * colsB, cudaMemcpyHostToDevice);

    // Perform matrix multiplication using cuBLAS cublasSgemm
    const float alpha = 1.0f;
    const float beta = 0.0f;
    ...

    // Print the result of the matrix multiplication
    float* h_C = (float*)malloc(sizeof(float) * rowsA * colsB);
    cudaMemcpy(h_C, d_C, sizeof(float) * rowsA * colsB, cudaMemcpyDeviceToHost);

    printf("A:\n");
    printMatrix(h_A, M, K);
    printf("\n"); 
    
    printf("B:\n");
    printMatrix(h_B, K, N);
    printf("\n"); 

    printf("C before ReLU activation:\n"); 
    printMatrix(h_C, M, N);
    printf("\n"); 

    // Create a cuDNN tensor descriptor for the output of the matrix multiplication
    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rowsA, colsB);

    // Create a cuDNN activation descriptor for the ReLU activation function
    cudnnActivationDescriptor_t activationDesc;
    cudnnCreateActivationDescriptor(&activationDesc);
    cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    // Apply the ReLU activation function using cuDNN
    ...

    // Print the result after the ReLU activation
    cudaMemcpy(h_C, d_C, sizeof(float) * rowsA * colsB, cudaMemcpyDeviceToHost);

    printf("C after ReLU activation:\n");  
    printMatrix(h_C, M, N);

    // Free the memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}