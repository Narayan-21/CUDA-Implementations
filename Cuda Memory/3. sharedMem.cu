// 3. Shared memory variable - Shared Memory, Block scope, and lifetime grid

#include <stdio.h>

__global__ void kernelShared() {
    __shared__ int sharedVar;  // shared by all threads in a block

    if (threadIdx.x == 0) sharedVar = 42;
    __syncthreads();  // ensure all threads see the update

    printf("Thread %d sees sharedVar=%d\n", threadIdx.x, sharedVar);
}

int main() {
    kernelShared<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
