#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include "../include/image_processor.h"

// Display usage information
void printUsage(const char* programName) {
    std::cout << "GPU-Accelerated Image Processing" << std::endl;
    std::cout << "Usage: " << programName << " [input_image] [filter_type] [output_image]" << std::endl;
    std::cout << "Filter types:" << std::endl;
    std::cout << "  gaussian  - Apply Gaussian blur filter" << std::endl;
    std::cout << "  sobel     - Apply Sobel edge detection filter" << std::endl;
    std::cout << "  sharpen   - Apply sharpening filter" << std::endl;
    std::cout << "  emboss    - Apply emboss filter" << std::endl;
    std::cout << "  benchmark - Run all filters and measure performance" << std::endl;
}

// Run benchmark for all filters
void runBenchmark(ImageProcessor& processor, const std::string& inputImage, const std::string& outputPrefix) {
    std::vector<TimingInfo> timings;
    
    // Load the image
    if (!processor.loadImage(inputImage)) {
        return;
    }
    
    std::cout << "\n=== Running Performance Benchmark ===" << std::endl;
    
    // Run Gaussian blur filter
    std::cout << "\nApplying Gaussian Blur filter..." << std::endl;
    timings.push_back(processor.applyGaussianBlur());
    processor.saveImage(outputPrefix + "_gaussian.jpg");
    
    // Run Sobel filter
    std::cout << "\nApplying Sobel filter..." << std::endl;
    timings.push_back(processor.applySobelFilter());
    processor.saveImage(outputPrefix + "_sobel.jpg");
    
    // Run Sharpening filter
    std::cout << "\nApplying Sharpening filter..." << std::endl;
    timings.push_back(processor.applySharpeningFilter());
    processor.saveImage(outputPrefix + "_sharpen.jpg");
    
    // Run Emboss filter
    std::cout << "\nApplying Emboss filter..." << std::endl;
    timings.push_back(processor.applyEmbossFilter());
    processor.saveImage(outputPrefix + "_emboss.jpg");
    
    // Print summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << std::setw(15) << "Filter" << std::setw(15) << "CPU Time (s)" << std::setw(15) << "GPU Time (s)" << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<std::string> filterNames = {"Gaussian Blur", "Sobel", "Sharpening", "Emboss"};
    
    for (size_t i = 0; i < timings.size(); i++) {
        std::cout << std::setw(15) << filterNames[i] 
                  << std::setw(15) << std::fixed << std::setprecision(6) << timings[i].cpuTime
                  << std::setw(15) << std::fixed << std::setprecision(6) << timings[i].gpuTime
                  << std::setw(15) << std::fixed << std::setprecision(2) << timings[i].speedup << "x" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Check for CUDA devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Print device information
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    }
    
    // Check command line arguments
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    std::string inputImage = argv[1];
    std::string filterType = argv[2];
    std::string outputImage = argc > 3 ? argv[3] : "output.jpg";
    
    // Create image processor
    ImageProcessor processor;
    
    // Check if running benchmark
    if (filterType == "benchmark") {
        runBenchmark(processor, inputImage, "results/output");
        return 0;
    }
    
    // Load the image
    if (!processor.loadImage(inputImage)) {
        return 1;
    }
    
    // Apply the selected filter
    TimingInfo timing;
    
    if (filterType == "gaussian") {
        std::cout << "Applying Gaussian Blur filter..." << std::endl;
        timing = processor.applyGaussianBlur();
    } else if (filterType == "sobel") {
        std::cout << "Applying Sobel filter..." << std::endl;
        timing = processor.applySobelFilter();
    } else if (filterType == "sharpen") {
        std::cout << "Applying Sharpening filter..." << std::endl;
        timing = processor.applySharpeningFilter();
    } else if (filterType == "emboss") {
        std::cout << "Applying Emboss filter..." << std::endl;
        timing = processor.applyEmbossFilter();
    } else {
        std::cerr << "Unknown filter type: " << filterType << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Save the processed image
    if (!processor.saveImage(outputImage)) {
        return 1;
    }
    
    return 0;
} 