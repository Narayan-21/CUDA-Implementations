#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void kernel(int *c) {
    *c = (*c) * (*c);
}

int main(void) {
    // Host & device pointers
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Running on GPU: %s\n", prop.name);

    int *h_c, *d_c;
    int value = 5;

    h_c = (int*)malloc(sizeof(int));
    *h_c = value;

    // allocate memory on device
    cudaMalloc((void**)&d_c, sizeof(int));

    // copy data from host to device
    cudaMemcpy(d_c, h_c, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid_size(1);
    dim3 block_size(1);

    kernel<<<grid_size, block_size>>>(d_c);

    cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result = %d\n", *h_c);
    cudaFree(d_c);
    free(h_c);
    return 0;
}