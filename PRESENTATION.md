# GPU-Accelerated Image Processing

## Project Presentation

### Introduction

This project demonstrates the application of GPU computing for accelerating image processing tasks. Using NVIDIA's CUDA platform, we implement several common image filters and compare their performance between CPU and GPU implementations to showcase the power of parallel computing.

### Project Goals

1. Demonstrate the advantages of GPU computing for image processing tasks
2. Implement and compare various image processing filters
3. Provide performance benchmarks between CPU and GPU implementations
4. Create a framework for future expansion of GPU-accelerated image processing

### Implementation Details

#### Architecture

The project is built with a modular architecture:

1. **Image Processing Class**: Handles loading/saving images and executing filters
2. **CUDA Kernels**: Implements parallelized filter algorithms
3. **CPU Comparison Functions**: Sequential implementations for benchmarking
4. **Benchmarking Framework**: Measures and reports performance differences

#### Filters Implemented

1. **Gaussian Blur**: Smooths images by convolving with a Gaussian kernel
   - Filter radius is configurable
   - Uses separable implementation for better performance

2. **Sobel Edge Detection**: Identifies edges by calculating intensity gradients
   - Computes horizontal and vertical gradients
   - Combines them to find edge magnitude

3. **Sharpening Filter**: Enhances edges while preserving overall appearance
   - Uses a 3x3 kernel for center enhancement and neighbor reduction

4. **Emboss Filter**: Creates a 3D-like effect with directional lighting
   - Applies directional convolution
   - Adds a mid-gray offset for proper visualization

### CUDA Implementation

#### Thread Organization

- Each pixel is processed by a separate CUDA thread
- Threads are organized in 2D blocks (16x16) to match the 2D nature of image data
- The grid of blocks covers the entire image

```cuda
// Calculate block and grid dimensions
dim3 threads(16, 16);
dim3 blocks((width + threads.x - 1) / threads.x, 
            (height + threads.y - 1) / threads.y);
```

#### Memory Management

- Image data is transferred between host and device memory
- Uses pinned memory for faster transfers when appropriate
- Properly manages device memory allocation/deallocation

#### Error Handling

- All CUDA API calls are checked for errors
- Proper cleanup on error conditions

### Performance Results

Performance varies by image size and GPU capabilities, but typical results show:

| Filter         | CPU Time (s) | GPU Time (s) | Speedup |
|----------------|--------------|--------------|---------|
| Gaussian Blur  | 0.245        | 0.012        | 20.4x   |
| Sobel          | 0.187        | 0.008        | 23.4x   |
| Sharpening     | 0.172        | 0.007        | 24.6x   |
| Emboss         | 0.169        | 0.007        | 24.1x   |

*Note: These are sample results. Actual performance will vary based on hardware and image size.*

### Visual Results

#### Original Image
![Original Image](results/original.jpg)

#### Gaussian Blur
![Gaussian Blur](results/output_gaussian.jpg)

#### Sobel Edge Detection
![Sobel Edge Detection](results/output_sobel.jpg)

#### Sharpening Filter
![Sharpening Filter](results/output_sharpen.jpg)

#### Emboss Filter
![Emboss Filter](results/output_emboss.jpg)

### Challenges and Solutions

1. **Memory Management**: Ensuring proper allocation and deallocation of device memory
   - Solution: Implemented RAII-style resource management

2. **Error Handling**: Robust handling of CUDA errors
   - Solution: Comprehensive error checking for all CUDA operations

3. **Performance Optimization**: Maximizing GPU utilization
   - Solution: Optimal thread organization and minimizing host-device transfers

### Conclusions

1. GPU acceleration provides significant performance improvements for image processing
2. Different filters benefit differently from parallelization based on algorithm characteristics
3. Proper implementation of parallel algorithms is essential for realizing the full potential of GPUs

### Future Work

1. Implement additional filters (e.g., median filter, bilateral filter)
2. Explore shared memory optimization for better performance
3. Add support for real-time video processing
4. Compare performance across different GPU architectures

### References

1. NVIDIA CUDA Programming Guide
2. Digital Image Processing (Gonzalez & Woods)
3. OpenCV Documentation 