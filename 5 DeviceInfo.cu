//  Get cuda device info

#include <stdio.h>
#include <cuda.h>
#include "cuda_error_handler.h"
#include <cuda_runtime.h>

void printDeviceProperties(const cudaDeviceProp &prop, int deviceId) {
    printf("========== Device %d: %s ==========\n", deviceId, prop.name);
    printf("Total Global Memory:                                                           %zu bytes\n", prop.totalGlobalMem);
    printf("Shared Memory per Block:                                                       %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per Block:                                                           %d\n", prop.regsPerBlock);
    printf("Warp Size:                                                                     %d\n", prop.warpSize);
    printf("Memory Pitch:                                                                  %zu bytes\n", prop.memPitch);
    printf("Max Threads per Block:                                                         %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dimension:                                                         [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Size:                                                                 [%d, %d, %d]\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total Constant Memory:                                                         %zu bytes\n", prop.totalConstMem);
    printf("Compute Capability:                                                            %d.%d\n", prop.major, prop.minor);
    printf("Clock Rate:                                                                    %d kHz\n", prop.clockRate);
    printf("Texture Alignment:                                                             %zu bytes\n", prop.textureAlignment);
    printf("Device Overlap:                                                                %d\n", prop.deviceOverlap);
    printf("Multiprocessor Count(Number of streaming multiprocessors (SM)):                %d\n", prop.multiProcessorCount);
    printf("Kernel Execution Timeout Enabled:                                              %d\n", prop.kernelExecTimeoutEnabled);
    printf("Integrated GPU (shares sys mem):                                               %d\n", prop.integrated);
    printf("Can Map Host Memory:                                                           %d\n", prop.canMapHostMemory);
    printf("Compute Mode:                                                                  %d\n", prop.computeMode);
    printf("Max Texture 1D:                                                                %d\n", prop.maxTexture1D);
    printf("Max Texture 2D:                                                                [%d, %d]\n",
           prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("Max Texture 3D:                                                                [%d, %d, %d]\n",
           prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("Concurrent Kernels:                                                            %d\n", prop.concurrentKernels);
    printf("PCI Bus ID:                                                                    %d\n", prop.pciBusID);
    printf("PCI Device ID:                                                                 %d\n", prop.pciDeviceID);
    printf("PCI Domain ID:                                                                 %d\n", prop.pciDomainID);
    printf("Unified Addressing:                                                            %d\n", prop.unifiedAddressing);
    printf("Memory Clock Rate:                                                             %d kHz\n", prop.memoryClockRate);
    printf("Memory Bus Width:                                                              %d bits\n", prop.memoryBusWidth);
    printf("L2 Cache Size:                                                                 %d bytes\n", prop.l2CacheSize);
    printf("Max Threads per Multiprocessor:                                                %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Async Engine Count:                                                            %d\n", prop.asyncEngineCount);
    printf("ECC Enabled:                                                                   %d\n", prop.ECCEnabled);
    printf("TCC Driver:                                                                    %d\n", prop.tccDriver);
    printf("Unified Addressing:                                                            %d\n", prop.unifiedAddressing);
    printf("=============================================\n\n");
}


int main(void) {
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printDeviceProperties(prop, i);
    }

    return 0;
}