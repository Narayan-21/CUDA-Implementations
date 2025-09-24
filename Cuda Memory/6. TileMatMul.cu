// Tiled matrix multiplication using shared memory

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 2

__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int ROW = by * TILE_WIDTH + ty;
    int COL = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // Colaborative loading of M and N tiles into the shared memory
        Mds[ty][tx] = M[ROW*Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + COL];

        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        __syncthreads();
    }
    P[ROW * Width + COL] = Pvalue;
};

void randomInit(float* data, int size) {
    for (int i=0; i<size; i++)
        data[i] = rand() / (float)RAND_MAX;
};

int main() {
    int Width = 4; 
    int size = Width * Width;
    size_t bytes = size * sizeof(float);

    float* h_M = (float*)malloc(bytes);
    float* h_N = (float*)malloc(bytes);
    float* h_P = (float*)malloc(bytes);

    randomInit(h_M, size);
    randomInit(h_N, size);

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);

    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / dimBlock.x, Width / dimBlock.y);

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);

    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
 printf("\nMatrix M:\n");
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%10.3f ", h_M[i * Width + j]);
        }
        printf("\n");
    }

    printf("\nMatrix N:\n");
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%10.3f ", h_N[i * Width + j]);
        }
        printf("\n");
    }

    printf("\nMatrix P = M x N:\n");
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%10.3f ", h_P[i * Width + j]);
        }
        printf("\n");
    }

    free(h_M); free(h_N); free(h_P); 
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P); 
    return 0;
}   