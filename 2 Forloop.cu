// Cuda - For loop
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void increment_gpu(int *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = a[i] + 1;
    };
};

int main(void) {
    const int N = 10;
    int h_a[N];

    for (int i =0; i < N; i++) {
        h_a[i] = i;
    }

    int *d_a;
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Launch enough threads to cover the N
    int threadsPerBlock = 3;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; // ceil(N / threadsPerBlock)

    dim3 grid_size(blocks);
    dim3 block_size(threadsPerBlock);

    increment_gpu<<<grid_size, block_size>>>(d_a, N);

    cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result after increment: \n");
    for (int i=0; i<N; i++) {
        printf("%d", h_a[i]);
    }
    printf("\n");
    cudaFree(d_a);
    return 0;
}