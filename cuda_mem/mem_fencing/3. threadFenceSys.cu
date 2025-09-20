// __threadfence_system() - visibility to host and device.

#include <stdio.h>

__global__ void threadFenceSystem(int *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = tid;
    __threadfence_system();
};

int main() {
    int *d_data;
    int h_data[256];

    cudaMalloc(&d_data, 256*sizeof(int));
    threadFenceSystem<<<1, 256>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, 256*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i =0; i<256; i++)
        printf("Data[%d] = %d\n", i, h_data[i]);
    
    cudaFree(d_data);
    return 0;
}