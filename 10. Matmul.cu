#include <stdio.h>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col]; // (same row + changing col) * (changing row + same col)
        }
        P[row * Width + col] = Pvalue;
    }
}

int main() {
    int width = 4;
    int size = width * width * sizeof(float);

    float *h_M = (float *)malloc(size);
    float *h_N = (float *)malloc(size);
    float *h_P = (float *)malloc(size);

    for (int i=0; i<width*width; i++) {
        h_M[i] = 1.0f;
        h_N[i] = 2.0f;
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(16,16,1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y, 1);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    printf("Result matrix P:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", h_P[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}