#include "cuda_error_handler.h"

void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err)
                  << " in " << file
                  << " at line " << line
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}