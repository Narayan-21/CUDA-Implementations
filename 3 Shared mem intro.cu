#include <stdio.h>

__global__ void kernel(int* in, int* out, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Allocating a shared array (size = blockDim.x, one element per thread)
    __shared__ int shared_array[1024];

    if (i < N) {
        shared_array[threadIdx.x] = in[1];
        __syncthreads(); // ensures all threads finish copying
        shared_array[threadIdx.x] *= 2;
        __syncthreads();
        out[i] = shared_array[threadIdx.x];
    }
}

int main() {
    long long N = 100000000;
    size_t size = N * sizeof(int);

    int *h_in = (int*)malloc(size);
    int *h_out = (int*)malloc(size);

    for (int i=0; i<N; i++) h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("Input -> Output\n");
    for (int i=0; i<N; i++) {
        printf("%d -> %d\n", h_in[i], h_out[i]);
    }

    // Free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

// cudaDeviceSynchronize - Forces host to wait on the completion of the kernel
// there is always an implicit barrier between kernels hence the grid from the 2nd kernel launch will not be scheduled to be executed on the device until the first kernel has completed its execution.
