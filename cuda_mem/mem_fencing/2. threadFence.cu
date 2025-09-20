// 2.__threadfence() - global memory visibility across all threads on a device

#include <stdio.h>

__device__ volatile int flag = 0;
__device__ int data = 0;

// This will work since the producer-consumer are within the same block 
__global__ void producerConsumer() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        // producer thread
        data = 123;         // Writes to global memory
        __threadfence();    // Ensures all the blocks see this write
        flag = 1;           // Signal consumer that data is ready
        printf("Producer (block %d): Data written and flag set\n", blockIdx.x);
    } else if (tid == 1) {
        while (flag==0); // Busy-wait
        printf("Consumer sees data = %d\n", data);
    }
}

__global__ void producerConsumerDifferentBlock() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        data = 246;
        __threadfence();
        flag = 2;
        printf("Producer (block %d): Data written and flag set\n", blockIdx.x);
    } else if (tid == 256) {
        int count = 0;
        while (flag != 2 && count < 1000000) {
            count++;
        }
        if (flag == 2) {
            printf("Consumer (block %d) sees data = %d after %d iterations\n", blockIdx.x, data, count);
        } else {
            printf("Consumer (block %d) timed out waiting for flag\n", blockIdx.x);
        }
    }
}

int main() {

    int zero = 0;
    cudaMemcpyToSymbol(flag, &zero, sizeof(int));
    cudaMemcpyToSymbol(data, &zero, sizeof(int));
    
    producerConsumer<<<2, 256>>>();
    cudaDeviceSynchronize();

    producerConsumerDifferentBlock<<<2,256>>>();
    cudaDeviceSynchronize();

    return 0;
}
