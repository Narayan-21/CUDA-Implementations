#include <stdio.h>
#include <cuda_runtime.h>

// Number of channels in RGB image
#define CHANNELS 3

// CUDA kernel: convert RGB image to grayscale
__global__
void colorToGrayscaleConversion(unsigned char *Pout,
                                unsigned char *Pin,
                                int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Linear index for grayscale image
        int grayOffset = row * width + col;

        // Linear index for RGB image (3 channels per pixel)
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];     // Red
        unsigned char g = Pin[rgbOffset + 1]; // Green
        unsigned char b = Pin[rgbOffset + 2]; // Blue

        // Weighted sum to convert to grayscale
        Pout[grayOffset] = (unsigned char)(0.21f * r +
                                           0.71f * g +
                                           0.07f * b);
    }
}

int main() {
    // Example dimensions (1500 x 2000 image)
    int width = 2000;
    int height = 1500;
    int numPixels = width * height;

    // Host memory allocation
    unsigned char *h_in  = (unsigned char*)malloc(numPixels * CHANNELS);
    unsigned char *h_out = (unsigned char*)malloc(numPixels);

    // (Here you would normally load image data into h_in)
    // For demo: just fill with random data
    for (int i = 0; i < numPixels * CHANNELS; i++) {
        h_in[i] = (unsigned char)(rand() % 256);
    }

    // Device memory allocation
    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in,  numPixels * CHANNELS);
    cudaMalloc((void**)&d_out, numPixels);

    // Copy input image to device
    cudaMemcpy(d_in, h_in, numPixels * CHANNELS, cudaMemcpyHostToDevice);

    // Configure blocks and grids
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y,
                 1);

    // Launch kernel
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(d_out, d_in, width, height);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, numPixels, cudaMemcpyDeviceToHost);

    // (Here you would normally save h_out as a grayscale image)
    printf("Conversion done! Example output pixel[0] = %d\n", h_out[0]);

    // Free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
