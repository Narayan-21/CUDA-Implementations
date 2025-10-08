// Tiled MatMul with ThreadCoarsening optimization

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    for (int c=0; c<COARSE_FACTOR; ++c){
        Pvalue[c] = 0.0f;
    };

    for (int ph=0; ph < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int mCol = ph * TILE_WIDTH + tx;
        if (row < width && mCol < width) {
            Mds[ty][tx] = M[row*width + mCol];
        } else {
            Mds[ty][tx] = 0.0f;
        }
        // Load tiles of N for each coarsened element and compute
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * TILE_WIDTH;
            int nRow = ph * TILE_WIDTH + ty;
            
            if (nRow < width && col < width){
                Nds[ty][tx] = N[nRow * width + col];
            } else{
                Nds[ty][tx] = 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k)
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            
            __syncthreads();
        };
    }
    if (row < width) {
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * TILE_WIDTH;
            if (col < width)
                P[row * width + col] = Pvalue[c];
        }
    }
};

int main() {
    int width = 1024;
    size_t size = width * width * sizeof(float);
    float *M = (float*)malloc(size);
    float *N = (float*)malloc(size);
    float *P = (float*)malloc(size);

    // Simple matrix initialization
    for (int i = 0; i < width*width; i++) {
        M[i] = 1.0f;
        N[i] = 1.0f;
    };

    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH* COARSE_FACTOR - 1) / (TILE_WIDTH* COARSE_FACTOR), 
    (width + TILE_WIDTH - 1) / (TILE_WIDTH)    
    );

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    bool correct = true;
    float expected = (float)width;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (P[i * width + j] != expected) {
                printf("Error at P[%d][%d]: expected %.1f, got %.1f\n", 
                       i, j, expected, P[i * width + j]);
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    
    if (correct) {
        printf("Success! All elements are %.1f\n", expected);
        printf("Sample values from P:\n");
        printf("P[0][0] = %.1f\n", P[0]);
        printf("P[511][511] = %.1f\n", P[511 * width + 511]);
        printf("P[1023][1023] = %.1f\n", P[1023 * width + 1023]);
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(M);
    free(N);
    free(P);
    
    return 0;

}