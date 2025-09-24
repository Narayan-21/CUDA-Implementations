// 5. Constant Memory - constant memory, grid scope, application lifetime
// Only the host can modify the constant variable.

#include <stdio.h>

__device__ __constant__ int constVar;   // constant in device memory (read-only)

__global__ void kernelConstant() {
    printf("Thread %d reads constVar=%d\n", threadIdx.x, constVar);
}

int main() {
    int hVal = 99;
    cudaMemcpyToSymbol(constVar, &hVal, sizeof(int)); // copy from host to constant memory

    kernelConstant<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
