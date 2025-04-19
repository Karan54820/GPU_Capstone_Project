#include "../include/image_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace cuda {

// Helper function to calculate block and thread dimensions
cudaError_t calculateBlockSize(int width, int height, dim3& blocks, dim3& threads) {
    // Use 16x16 thread blocks
    threads = dim3(16, 16);
    
    // Calculate grid dimensions
    blocks = dim3((width + threads.x - 1) / threads.x, 
                  (height + threads.y - 1) / threads.y);
    
    return cudaSuccess;
}

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output,
                                  int width, int height, int channels, int radius,
                                  float* kernel, int kernelSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Process each color channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Apply convolution with the Gaussian kernel
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                unsigned char pixel = input[(py * width + px) * channels + c];
                int kidx = (ky + radius) * kernelSize + (kx + radius);
                sum += pixel * kernel[kidx];
            }
        }
        
        // Write the result
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
    }
}

// CUDA kernel for Sobel filter
__global__ void sobelFilterKernel(unsigned char* input, unsigned char* output,
                                 int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Sobel operators
    const int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const int sobelY[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    
    // Process each color channel
    for (int c = 0; c < channels; c++) {
        int gx = 0;
        int gy = 0;
        
        // Apply convolution with the Sobel operators
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                unsigned char pixel = input[(py * width + px) * channels + c];
                
                gx += pixel * sobelX[ky + 1][kx + 1];
                gy += pixel * sobelY[ky + 1][kx + 1];
            }
        }
        
        // Calculate gradient magnitude
        int magnitude = min(255, max(0, static_cast<int>(sqrtf(gx * gx + gy * gy))));
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(magnitude);
    }
}

// CUDA kernel for sharpening filter
__global__ void sharpeningFilterKernel(unsigned char* input, unsigned char* output,
                                      int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Sharpening kernel
    const int kernel[3][3] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };
    
    // Process each color channel
    for (int c = 0; c < channels; c++) {
        int sum = 0;
        
        // Apply convolution with the sharpening kernel
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                unsigned char pixel = input[(py * width + px) * channels + c];
                sum += pixel * kernel[ky + 1][kx + 1];
            }
        }
        
        // Clamp to valid range
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(255, max(0, sum)));
    }
}

// CUDA kernel for emboss filter
__global__ void embossFilterKernel(unsigned char* input, unsigned char* output,
                                  int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Emboss kernel
    const int kernel[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    };
    
    // Process each color channel
    for (int c = 0; c < channels; c++) {
        int sum = 0;
        
        // Apply convolution with the emboss kernel
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                unsigned char pixel = input[(py * width + px) * channels + c];
                sum += pixel * kernel[ky + 1][kx + 1];
            }
        }
        
        // Add 128 to shift to middle range and clamp
        sum = min(255, max(0, sum + 128));
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
    }
}

// Wrapper function for launching Gaussian blur kernel
cudaError_t launchGaussianBlur(unsigned char* d_input, unsigned char* d_output,
                              int width, int height, int channels, int radius) {
    cudaError_t error;
    
    // Calculate launch configuration
    dim3 blocks, threads;
    error = calculateBlockSize(width, height, blocks, threads);
    if (error != cudaSuccess) {
        return error;
    }
    
    // Prepare Gaussian kernel on host
    int kernelSize = 2 * radius + 1;
    float* h_kernel = new float[kernelSize * kernelSize];
    float sigma = radius / 3.0f;
    float sum = 0.0f;
    
    // Compute 2D Gaussian kernel values
    for (int y = 0; y < kernelSize; y++) {
        for (int x = 0; x < kernelSize; x++) {
            int kx = x - radius;
            int ky = y - radius;
            h_kernel[y * kernelSize + x] = expf(-(kx * kx + ky * ky) / (2 * sigma * sigma));
            sum += h_kernel[y * kernelSize + x];
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        h_kernel[i] /= sum;
    }
    
    // Copy kernel to device
    float* d_kernel;
    error = cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    if (error != cudaSuccess) {
        delete[] h_kernel;
        return error;
    }
    
    error = cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_kernel);
        delete[] h_kernel;
        return error;
    }
    
    // Launch kernel
    gaussianBlurKernel<<<blocks, threads>>>(d_input, d_output, width, height, channels, radius, d_kernel, kernelSize);
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_kernel);
        delete[] h_kernel;
        return error;
    }
    
    // Clean up
    cudaFree(d_kernel);
    delete[] h_kernel;
    
    return cudaSuccess;
}

// Wrapper function for launching Sobel filter kernel
cudaError_t launchSobelFilter(unsigned char* d_input, unsigned char* d_output,
                            int width, int height, int channels) {
    // Calculate launch configuration
    dim3 blocks, threads;
    cudaError_t error = calculateBlockSize(width, height, blocks, threads);
    if (error != cudaSuccess) {
        return error;
    }
    
    // Launch kernel
    sobelFilterKernel<<<blocks, threads>>>(d_input, d_output, width, height, channels);
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    
    return cudaSuccess;
}

// Wrapper function for launching sharpening filter kernel
cudaError_t launchSharpeningFilter(unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, int channels) {
    // Calculate launch configuration
    dim3 blocks, threads;
    cudaError_t error = calculateBlockSize(width, height, blocks, threads);
    if (error != cudaSuccess) {
        return error;
    }
    
    // Launch kernel
    sharpeningFilterKernel<<<blocks, threads>>>(d_input, d_output, width, height, channels);
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    
    return cudaSuccess;
}

// Wrapper function for launching emboss filter kernel
cudaError_t launchEmbossFilter(unsigned char* d_input, unsigned char* d_output,
                             int width, int height, int channels) {
    // Calculate launch configuration
    dim3 blocks, threads;
    cudaError_t error = calculateBlockSize(width, height, blocks, threads);
    if (error != cudaSuccess) {
        return error;
    }
    
    // Launch kernel
    embossFilterKernel<<<blocks, threads>>>(d_input, d_output, width, height, channels);
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    
    return cudaSuccess;
}

} // namespace cuda 