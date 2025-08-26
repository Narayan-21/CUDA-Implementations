#ifndef CUDA_ERROR_HANDLER_H
#define CUDA_ERROR_HANDLER_H

#include <cuda_runtime.h>
#include <iostream>

// Declare the function
void HandleError(cudaError_t err, const char *file, int line);

// Define the macro here (so it expands properly in user files)
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#endif // CUDA_ERROR_HANDLER_H
