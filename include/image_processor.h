#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Struct to hold timing information for performance comparison
struct TimingInfo {
    float cpuTime;
    float gpuTime;
    float speedup;
};

// Image processor class
class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // Load an image from file
    bool loadImage(const std::string& filename);
    
    // Save the processed image to file
    bool saveImage(const std::string& filename);

    // Process image with different filters
    TimingInfo applyGaussianBlur(int radius = 5);
    TimingInfo applySobelFilter();
    TimingInfo applySharpeningFilter();
    TimingInfo applyEmbossFilter();

    // Get the processed image
    cv::Mat getProcessedImage() const;

private:
    cv::Mat originalImage;
    cv::Mat processedImage;
    
    // Allocate and free CUDA memory
    void allocateCudaMemory();
    void freeCudaMemory();
    
    // CUDA device memory pointers
    unsigned char* d_input;
    unsigned char* d_output;
    
    // Image dimensions
    int width;
    int height;
    int channels;
    size_t imageSize;
    
    // CPU implementation of filters for comparison
    void cpuGaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius);
    void cpuSobelFilter(unsigned char* input, unsigned char* output, int width, int height, int channels);
    void cpuSharpeningFilter(unsigned char* input, unsigned char* output, int width, int height, int channels);
    void cpuEmbossFilter(unsigned char* input, unsigned char* output, int width, int height, int channels);
}; 