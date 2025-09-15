#include <stdio.h>
#include <cuda_runtime.h>

#define BLUR_SIZE 1

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }
        out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
}

int main() {
    int width = 8;
    int height = 8;
    int imgSize = width * height;

    unsigned char *h_in = (unsigned char*)malloc(imgSize);
    unsigned char *h_out = (unsigned char*)malloc(imgSize);

    // Dummy input array
    for (int i=0; i<imgSize; i++) {
        h_in[i] = rand() % 256;
    }

    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, imgSize);
    cudaMalloc((void**)&d_out, imgSize);

    cudaMemcpy(d_in, h_in, imgSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    blurKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height);

    cudaMemcpy(h_out, d_out, imgSize, cudaMemcpyDeviceToHost);
    
    printf("Blurred image: \n");
    for (int i=0; i<height; i++){
        for (int j = 0; j<width; j++) {
            printf("%3d", h_out[i*width+j]);
        }
        printf("\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}