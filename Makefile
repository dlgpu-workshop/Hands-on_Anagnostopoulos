CC = nvcc
LDFLAGS = -lcublas -lcudnn -lnccl
INC = -I/usr/include
# INC = 
TARGET = cuda_simple cuda_simple_error_checking matmul work_done_solution cudnn_simple conv1d cublas_cudnn_solution ncclBcast ncclReduce ncclAllReduce addarrays addarraysNVTX

all: $(TARGET)
		
# Code for slides part 1

cuda_simple: cuda_simple.cu
	$(CC) $(CFLAGS) -o $@ $< 

cuda_simple_error_checking: cuda_simple_error_checking.cu
	$(CC) $(CFLAGS) -o $@ $< 

matmul: matmul.cu
	$(CC) $(CFLAGS) -o $@ $< -lcublas

work_done_solution: work_done_solution.cu
	$(CC) $(CFLAGS) -o $@ $< -lcublas

cudnn_simple: cudnn_simple.cu
	$(CC) $(CFLAGS) -o $@ $< -lcudnn

conv1d: conv1d.cu
	$(CC) $(CFLAGS) -o $@ $< -lcudnn

cublas_cudnn_solution: cublas_cudnn_solution.cu
	$(CC) $(CFLAGS) -o $@ $< -lcublas  -lcudnn

# Code for slides part 2
addarrays: addarrays.cu
	$(CC) -o $@ $^

addarraysNVTX: addarraysNVTX.cu
	$(CC) -o $@ $^ -lnvToolsExt

ncclBcast: ncclBcast.cu
	$(CC) -o $@ $^ -lnccl $(INC)

ncclReduce: ncclReduce.cu
	$(CC) -o $@ $^ -lnccl $(INC)

ncclAllReduce: ncclAllReduce.cu
	$(CC) -o $@ $^ -lnccl $(INC)


clean:
	rm -f $(TARGET)
