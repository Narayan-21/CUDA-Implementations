// 1. Automatic variables other than arrays - Stored in a Register - register mem, thread scope, grid lifetime

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernelRegister() {
    int x = threadIdx.x; // stored in a register
    printf("Thread %d has register val x=%d\n", threadIdx.x, x);
}

int main() {
    kernelRegister<<<1,4>>>();
    cudaDeviceSynchronize();
    return 0;
}