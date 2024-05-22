#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
 
__global__ void kernel(int *a) 
{
  int index = threadIdx.x;

  a[index] *= 2;
  printf("%d\t", a[index]);

}/*kernel*/
 

void print_vector(int *in, int n){

 printf("Random values in the vector before ncclBcast\n");
 for(int i=0; i < n; i++)
  printf("%d\t", in[i]);

 printf("\n");

}/*print_vector*/


int main(int argc, char* argv[]) {

  int data_size = 8 ;
  int nGPUs = 0;
  cudaGetDeviceCount(&nGPUs);
  // print a message about the number of GPUs
  printf("Number of GPUs: %d\n\n", nGPUs);
    
  
  int *DeviceList = (int *) malloc (nGPUs     * sizeof(int));
  int *data       = (int*)  malloc (data_size * sizeof(int));
  int **d_data    = (int**) malloc (nGPUs     * sizeof(int*));
  
  for(int i = 0; i < nGPUs; i++)
      DeviceList[i] = i;
  
  /*Initializing NCCL with Multiples Devices per Thread*/
  ncclComm_t* comms = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nGPUs);  
  cudaStream_t* s   = (cudaStream_t*)malloc(sizeof(cudaStream_t)* nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList);
  
  /* initialize data vector with random values from 0 to 100 */
  for (int i = 0; i < data_size; i++)
      data[i] = rand() % 100;    

  /* print the data */
  print_vector(data, data_size);
      
  for(int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      cudaStreamCreate(&s[g]);
      cudaMalloc(&d_data[g], data_size * sizeof(int));
     
      if(g == 0)  /*Copy from Host to Device*/
         cudaMemcpy(d_data[g], data, data_size * sizeof(int), cudaMemcpyHostToDevice);
  }
        
  ncclGroupStart();
 
  		for(int g = 0; g < nGPUs; g++) {
  	  	    cudaSetDevice(DeviceList[g]);
    	  	ncclBcast(d_data[g], data_size, ncclInt, 0, comms[g], s[g]); /*Broadcasting it to all*/
  		}

  ncclGroupEnd();       

  for (int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      printf("\nThis is device %d\n", g);
      kernel <<< 1 , data_size >>> (d_data[g]);/*Call the CUDA Kernel: The code multiple the vector position per 2 on GPUs*/
      cudaDeviceSynchronize();             
  }

  printf("\n");

  for (int g = 0; g < nGPUs; g++) { /*Synchronizing CUDA Streams*/
      cudaSetDevice(DeviceList[g]);
      cudaStreamSynchronize(s[g]);
  }
 
  for(int g = 0; g < nGPUs; g++) {  /*Destroy CUDA Streams*/
      cudaSetDevice(DeviceList[g]);
      cudaStreamDestroy(s[g]);
  }

  for(int g = 0; g < nGPUs; g++)    /*Finalizing NCCL*/
     ncclCommDestroy(comms[g]);
  
  /*Freeing memory*/
  free(s);
  free(data); 
  free(DeviceList);

  cudaFree(d_data);

  return 0;

}/*main*/
