#include "../include/image_processor.h"
#include "../include/image_kernels.cuh"
#include <iostream>
#include <chrono>

// Constructor
ImageProcessor::ImageProcessor() : d_input(nullptr), d_output(nullptr), width(0), height(0), channels(0), imageSize(0) {
}

// Destructor
ImageProcessor::~ImageProcessor() {
    freeCudaMemory();
}

// Load image from file
bool ImageProcessor::loadImage(const std::string& filename) {
    originalImage = cv::imread(filename, cv::IMREAD_UNCHANGED);
    
    if (originalImage.empty()) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return false;
    }
    
    width = originalImage.cols;
    height = originalImage.rows;
    channels = originalImage.channels();
    imageSize = width * height * channels;
    
    std::cout << "Loaded image: " << filename << std::endl;
    std::cout << "Dimensions: " << width << "x" << height << " with " << channels << " channels" << std::endl;
    
    // Allocate CUDA memory
    allocateCudaMemory();
    
    return true;
}

// Save image to file
bool ImageProcessor::saveImage(const std::string& filename) {
    if (processedImage.empty()) {
        std::cerr << "No processed image to save" << std::endl;
        return false;
    }
    
    bool success = cv::imwrite(filename, processedImage);
    if (success) {
        std::cout << "Image saved to: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image to: " << filename << std::endl;
    }
    
    return success;
}

// Get the processed image
cv::Mat ImageProcessor::getProcessedImage() const {
    return processedImage;
}

// Allocate CUDA memory
void ImageProcessor::allocateCudaMemory() {
    // Free any previously allocated memory
    freeCudaMemory();
    
    // Allocate memory on the device
    cudaError_t err = cudaMalloc(&d_input, imageSize);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input image: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&d_output, imageSize);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output image: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        d_input = nullptr;
        return;
    }
}

// Free CUDA memory
void ImageProcessor::freeCudaMemory() {
    if (d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
}

// Apply Gaussian blur filter
TimingInfo ImageProcessor::applyGaussianBlur(int radius) {
    TimingInfo timing;
    
    if (originalImage.empty()) {
        std::cerr << "No image loaded" << std::endl;
        return timing;
    }
    
    // Create output image
    processedImage = cv::Mat(height, width, originalImage.type());
    
    // Start CPU timer
    auto cpuStart = std::chrono::steady_clock::now();
    
    // CPU implementation of Gaussian blur
    cpuGaussianBlur(originalImage.data, processedImage.data, width, height, channels, radius);
    
    // End CPU timer
    auto cpuEnd = std::chrono::steady_clock::now();
    timing.cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count() / 1000.0f;
    
    // Copy original image to device
    cudaError_t err = cudaMemcpy(d_input, originalImage.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy input image to device: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Start GPU timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch Gaussian blur kernel
    err = cuda::launchGaussianBlur(d_input, d_output, width, height, channels, radius);
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch Gaussian blur kernel: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // End GPU timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuElapsedTime;
    cudaEventElapsedTime(&gpuElapsedTime, start, stop);
    timing.gpuTime = gpuElapsedTime / 1000.0f;
    
    // Copy result back to host
    err = cudaMemcpy(processedImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy output image to host: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Calculate speedup
    timing.speedup = timing.cpuTime / timing.gpuTime;
    
    std::cout << "Gaussian Blur Filter applied" << std::endl;
    std::cout << "CPU time: " << timing.cpuTime << " seconds" << std::endl;
    std::cout << "GPU time: " << timing.gpuTime << " seconds" << std::endl;
    std::cout << "Speedup: " << timing.speedup << "x" << std::endl;
    
    return timing;
}

// Apply Sobel filter
TimingInfo ImageProcessor::applySobelFilter() {
    TimingInfo timing;
    
    if (originalImage.empty()) {
        std::cerr << "No image loaded" << std::endl;
        return timing;
    }
    
    // Create output image
    processedImage = cv::Mat(height, width, originalImage.type());
    
    // Start CPU timer
    auto cpuStart = std::chrono::steady_clock::now();
    
    // CPU implementation of Sobel filter
    cpuSobelFilter(originalImage.data, processedImage.data, width, height, channels);
    
    // End CPU timer
    auto cpuEnd = std::chrono::steady_clock::now();
    timing.cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count() / 1000.0f;
    
    // Copy original image to device
    cudaError_t err = cudaMemcpy(d_input, originalImage.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy input image to device: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Start GPU timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch Sobel filter kernel
    err = cuda::launchSobelFilter(d_input, d_output, width, height, channels);
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch Sobel filter kernel: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // End GPU timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuElapsedTime;
    cudaEventElapsedTime(&gpuElapsedTime, start, stop);
    timing.gpuTime = gpuElapsedTime / 1000.0f;
    
    // Copy result back to host
    err = cudaMemcpy(processedImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy output image to host: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Calculate speedup
    timing.speedup = timing.cpuTime / timing.gpuTime;
    
    std::cout << "Sobel Filter applied" << std::endl;
    std::cout << "CPU time: " << timing.cpuTime << " seconds" << std::endl;
    std::cout << "GPU time: " << timing.gpuTime << " seconds" << std::endl;
    std::cout << "Speedup: " << timing.speedup << "x" << std::endl;
    
    return timing;
}

// Apply Sharpening filter
TimingInfo ImageProcessor::applySharpeningFilter() {
    TimingInfo timing;
    
    if (originalImage.empty()) {
        std::cerr << "No image loaded" << std::endl;
        return timing;
    }
    
    // Create output image
    processedImage = cv::Mat(height, width, originalImage.type());
    
    // Start CPU timer
    auto cpuStart = std::chrono::steady_clock::now();
    
    // CPU implementation of Sharpening filter
    cpuSharpeningFilter(originalImage.data, processedImage.data, width, height, channels);
    
    // End CPU timer
    auto cpuEnd = std::chrono::steady_clock::now();
    timing.cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count() / 1000.0f;
    
    // Copy original image to device
    cudaError_t err = cudaMemcpy(d_input, originalImage.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy input image to device: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Start GPU timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch Sharpening filter kernel
    err = cuda::launchSharpeningFilter(d_input, d_output, width, height, channels);
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch Sharpening filter kernel: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // End GPU timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuElapsedTime;
    cudaEventElapsedTime(&gpuElapsedTime, start, stop);
    timing.gpuTime = gpuElapsedTime / 1000.0f;
    
    // Copy result back to host
    err = cudaMemcpy(processedImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy output image to host: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Calculate speedup
    timing.speedup = timing.cpuTime / timing.gpuTime;
    
    std::cout << "Sharpening Filter applied" << std::endl;
    std::cout << "CPU time: " << timing.cpuTime << " seconds" << std::endl;
    std::cout << "GPU time: " << timing.gpuTime << " seconds" << std::endl;
    std::cout << "Speedup: " << timing.speedup << "x" << std::endl;
    
    return timing;
}

// Apply Emboss filter
TimingInfo ImageProcessor::applyEmbossFilter() {
    TimingInfo timing;
    
    if (originalImage.empty()) {
        std::cerr << "No image loaded" << std::endl;
        return timing;
    }
    
    // Create output image
    processedImage = cv::Mat(height, width, originalImage.type());
    
    // Start CPU timer
    auto cpuStart = std::chrono::steady_clock::now();
    
    // CPU implementation of Emboss filter
    cpuEmbossFilter(originalImage.data, processedImage.data, width, height, channels);
    
    // End CPU timer
    auto cpuEnd = std::chrono::steady_clock::now();
    timing.cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count() / 1000.0f;
    
    // Copy original image to device
    cudaError_t err = cudaMemcpy(d_input, originalImage.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy input image to device: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Start GPU timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch Emboss filter kernel
    err = cuda::launchEmbossFilter(d_input, d_output, width, height, channels);
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch Emboss filter kernel: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // End GPU timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuElapsedTime;
    cudaEventElapsedTime(&gpuElapsedTime, start, stop);
    timing.gpuTime = gpuElapsedTime / 1000.0f;
    
    // Copy result back to host
    err = cudaMemcpy(processedImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy output image to host: " << cudaGetErrorString(err) << std::endl;
        return timing;
    }
    
    // Calculate speedup
    timing.speedup = timing.cpuTime / timing.gpuTime;
    
    std::cout << "Emboss Filter applied" << std::endl;
    std::cout << "CPU time: " << timing.cpuTime << " seconds" << std::endl;
    std::cout << "GPU time: " << timing.gpuTime << " seconds" << std::endl;
    std::cout << "Speedup: " << timing.speedup << "x" << std::endl;
    
    return timing;
}

// CPU implementation of Gaussian blur
void ImageProcessor::cpuGaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius) {
    // Create Gaussian kernel
    int kernelSize = 2 * radius + 1;
    float* kernel = new float[kernelSize];
    float sigma = radius / 3.0f;
    float sum = 0.0f;
    
    // Compute Gaussian kernel values
    for (int i = 0; i < kernelSize; i++) {
        int x = i - radius;
        kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    // Create temporary buffer for horizontal pass
    unsigned char* temp = new unsigned char[width * height * channels];
    
    // Horizontal pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                for (int k = -radius; k <= radius; k++) {
                    int px = std::min(std::max(x + k, 0), width - 1);
                    sum += input[(y * width + px) * channels + c] * kernel[k + radius];
                }
                
                temp[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
            }
        }
    }
    
    // Vertical pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                for (int k = -radius; k <= radius; k++) {
                    int py = std::min(std::max(y + k, 0), height - 1);
                    sum += temp[(py * width + x) * channels + c] * kernel[k + radius];
                }
                
                output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
            }
        }
    }
    
    delete[] kernel;
    delete[] temp;
}

// CPU implementation of Sobel filter
void ImageProcessor::cpuSobelFilter(unsigned char* input, unsigned char* output, int width, int height, int channels) {
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
    
    // Apply Sobel operator
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int gx = 0;
                int gy = 0;
                
                // Apply convolution
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int py = std::min(std::max(y + ky, 0), height - 1);
                        
                        unsigned char pixel = input[(py * width + px) * channels + c];
                        
                        gx += pixel * sobelX[ky + 1][kx + 1];
                        gy += pixel * sobelY[ky + 1][kx + 1];
                    }
                }
                
                // Calculate gradient magnitude
                int magnitude = std::min(255, std::max(0, static_cast<int>(sqrtf(gx * gx + gy * gy))));
                output[(y * width + x) * channels + c] = static_cast<unsigned char>(magnitude);
            }
        }
    }
}

// CPU implementation of Sharpening filter
void ImageProcessor::cpuSharpeningFilter(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    // Sharpening kernel
    const int kernel[3][3] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };
    
    // Apply sharpening filter
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                
                // Apply convolution
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int py = std::min(std::max(y + ky, 0), height - 1);
                        
                        unsigned char pixel = input[(py * width + px) * channels + c];
                        sum += pixel * kernel[ky + 1][kx + 1];
                    }
                }
                
                // Clamp to valid range
                output[(y * width + x) * channels + c] = static_cast<unsigned char>(std::min(255, std::max(0, sum)));
            }
        }
    }
}

// CPU implementation of Emboss filter
void ImageProcessor::cpuEmbossFilter(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    // Emboss kernel
    const int kernel[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    };
    
    // Apply emboss filter
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                
                // Apply convolution
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int py = std::min(std::max(y + ky, 0), height - 1);
                        
                        unsigned char pixel = input[(py * width + px) * channels + c];
                        sum += pixel * kernel[ky + 1][kx + 1];
                    }
                }
                
                // Add 128 to shift to middle range and clamp
                sum = std::min(255, std::max(0, sum + 128));
                output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
            }
        }
    }
} 