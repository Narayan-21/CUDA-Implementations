// Global Memory - global memory, grid scope and application lifetime

#include <stdio.h>

__device__ int globalVar;

__global__ void kernelGlobal() {
    globalVar = threadIdx.x;
}

int main() {
    kernelGlobal<<<1,8>>>();
    cudaDeviceSynchronize();

    int hVal;
    cudaMemcpyFromSymbol(&hVal, globalVar, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Final value of globalVar = %d\n", hVal);
    return 0;
}