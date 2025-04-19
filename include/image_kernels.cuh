#pragma once

#include <cuda_runtime.h>

// CUDA kernels for image processing
namespace cuda {

// Launch configuration helper
cudaError_t calculateBlockSize(int width, int height, dim3& blocks, dim3& threads);

// CUDA kernels for different filters
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, 
                                  int width, int height, int channels, int radius, 
                                  float* kernel, int kernelSize);

__global__ void sobelFilterKernel(unsigned char* input, unsigned char* output, 
                                 int width, int height, int channels);

__global__ void sharpeningFilterKernel(unsigned char* input, unsigned char* output, 
                                      int width, int height, int channels);

__global__ void embossFilterKernel(unsigned char* input, unsigned char* output, 
                                  int width, int height, int channels);

// Wrapper functions for kernel launches
cudaError_t launchGaussianBlur(unsigned char* d_input, unsigned char* d_output, 
                              int width, int height, int channels, int radius);

cudaError_t launchSobelFilter(unsigned char* d_input, unsigned char* d_output, 
                            int width, int height, int channels);

cudaError_t launchSharpeningFilter(unsigned char* d_input, unsigned char* d_output, 
                                 int width, int height, int channels);

cudaError_t launchEmbossFilter(unsigned char* d_input, unsigned char* d_output, 
                             int width, int height, int channels);

} // namespace cuda 