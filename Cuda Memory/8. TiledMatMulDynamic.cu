#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void MatrixMulDynamic(float* M, float* N, float* P, int Width) {
    extern __shared__ float Mds_Nds[];

    float* Mds = Mds_Nds;
    float* Nds = &Mds_Nds[TILE_WIDTH * TILE_WIDTH]; // Start right after the elements used by Mds

    int tx = threadIdx.x; int bx = blockIdx.x;
    int ty = threadIdx.y; int by = blockIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float PValue = 0.0f;
    for (int ph = 0; ph < ceil(Width / (float)TILE_WIDTH); ++ph) {
        if ((Row < Width) && (ph * TILE_WIDTH + tx) < Width)
            Mds[ty * TILE_WIDTH + tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        else
            Mds[ty * TILE_WIDTH + tx] = 0.0f;

        if ((ph * TILE_WIDTH + ty) < Width && Col < Width)
            Nds[ty * TILE_WIDTH + tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty * TILE_WIDTH + tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            PValue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
        }
        __syncthreads();
    }
    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = PValue;
    }
};

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (float)(rand() % 10);
    }
}

int main() {
    int Width = 64;
    int size = Width * Width;
    size_t memSize = size * sizeof(float);

    float* h_M = (float*)malloc(memSize);
    float* h_N = (float*)malloc(memSize);
    float* h_P = (float*)malloc(memSize);

    randomInit(h_M, size);
    randomInit(h_N, size);

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, memSize);
    cudaMalloc((void**)&d_N, memSize);
    cudaMalloc((void**)&d_P, memSize);

    cudaMemcpy(d_M, h_M, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, memSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Width + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    size_t sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    MatrixMulDynamic<<<dimGrid, dimBlock, sharedMemSize>>>(d_M, d_N, d_P, Width);
    cudaMemcpy(h_P, d_P, memSize, cudaMemcpyDeviceToHost);

    printf("Result matrix P (top-left 8x8 block):\n");
    for (int i = 0; i < 8 && i < Width; i++) {
        for (int j = 0; j < 8 && j < Width; j++) {
            printf("%6.1f ", h_P[i * Width + j]);
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