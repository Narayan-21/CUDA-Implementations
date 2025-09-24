// Cuda device selection based on some prop
// Task done - Choose the closest device from all available devices, device closest to compute capability of 1.3

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_handler.h"

int main(void) {
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("ID of the current CUDA devide: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;

    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);

    HANDLE_ERROR(cudaSetDevice(dev));
}