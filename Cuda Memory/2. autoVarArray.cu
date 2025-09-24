// 2. Automatic array variables - local memory, thread scope, grid lifetime
#include <stdio.h>

__global__ void kernelLocal() {
    int arr[3];
    arr[0] = threadIdx.x;
    printf("Thread %d has arr[0]=%d\n", threadIdx.x, arr[0]);
};

int main() {
    kernelLocal<<<1,4>>>();
    cudaDeviceSynchronize();
    return 0;
}