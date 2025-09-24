// Custom cuda error handler for memory allocation

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void add(int a, int b, int *c) {
    *c = a+b;
}

int main(void) {
    int c;
    int *dev_c;

    // GPU memory allocation
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

    // Launch Kernel
    add<<<1,1>>>(2,7,dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);

    cudaFree(dev_c);
    return 0;
}
