#include <stdio.h>
#include <cudnn.h>

int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t tensorDesc;

    // Initialize cuDNN
    cudnnCreate(&handle);

    // Create the tensor descriptor
    cudnnCreateTensorDescriptor(&tensorDesc);

    // Set the dimensions of the tensor
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    // Cleanup
    cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroy(handle);

    printf("cuDNN example completed successfully.\n");

    return 0;
}
