// Performance considerations between the naive matmul tiled implementation and Corner turning approach of matrix multiplication

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 64
#define N 16384 

__global__ void matmul_no_corner_turning(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float Cvalue = 0.0f;

    for (int t=0; t < n / TILE_SIZE; ++t) {
        As[threadIdx.y][threadIdx.x] =  A[row * n + (t * TILE_SIZE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }
    C[row * n + col] = Cvalue;
}

__global__ void matmul_with_corner_turning(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    for (int t = 0; t < n / TILE_SIZE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        // Load and transpose B ("corner turning")
        Bs[threadIdx.x][threadIdx.y] = B[(t * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += As[threadIdx.y][k] * Bs[threadIdx.x][k]; // Bs is transposed

        __syncthreads();
    }

    C[row * n + col] = Cvalue;
}

void init_matrix(float *mat, int n) {
    for (int i = 0; i < n * n; ++i)
        mat[i] = (float)(rand() % 100) / 10.0f;
}

float compare(const float *A, const float *B, int n) {
    float max_diff = 0;
    for (int i = 0; i < n * n; ++i)
        max_diff = fmaxf(max_diff, fabsf(A[i] - B[i]));
    return max_diff;
}

int main() {
    int size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C1 = (float *)malloc(size);
    float *h_C2 = (float *)malloc(size);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(N / TILE_SIZE, N / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Without Corner Turning ---
    cudaMemset(d_C, 0, size);
    cudaEventRecord(start);
    matmul_no_corner_turning<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_no_turn = 0;
    cudaEventElapsedTime(&time_no_turn, start, stop);
    cudaMemcpy(h_C1, d_C, size, cudaMemcpyDeviceToHost);

    // --- With Corner Turning ---
    cudaMemset(d_C, 0, size);
    cudaEventRecord(start);
    matmul_with_corner_turning<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_turn = 0;
    cudaEventElapsedTime(&time_turn, start, stop);
    cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost);

    float diff = compare(h_C1, h_C2, N);

    printf("\nMatrix size: %dx%d\n", N, N);
    printf("Without corner turning: %.3f ms\n", time_no_turn);
    printf("With corner turning   : %.3f ms\n", time_turn);
    printf("Speedup: %.2fx\n", time_no_turn / time_turn);
    printf("Max difference in results: %.6f\n\n", diff);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);

    return 0;
}
