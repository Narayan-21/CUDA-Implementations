// Mem fencing - Ensures memory ordering between threads or between a thread and the device memory.

// Example - __threadfence_block(): Ensures writes are visible to all threads in the same block.

// __threadfence() - global memory visibility across all threads on device
// __threadfence_block() - visibility within a block.
// __threadfence_system() - visibility to host and device.


#include <stdio.h>

__global__ void memThreadblockfence(int *data) {
    __shared__ int buffer;
    int tid = threadIdx.x;
    if (tid == 0) {
        buffer = 42;            // Thread 0 writes;
        __threadfence_block();  // Ensures all threads within a block sees the write
    }

    __syncthreads();
    if (tid == 1) {
        printf("Thread 1 sees: %d\n", buffer);
    };
};


int main() {
    memThreadblockfence<<<1,2>>>(nullptr);
    cudaDeviceSynchronize();
    return 0;
}