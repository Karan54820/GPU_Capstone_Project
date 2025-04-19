# GPU-Accelerated Image Processing

This project demonstrates the power of GPU computing for image processing operations using CUDA. It implements several commonly used image filters and provides a performance comparison between CPU and GPU implementations.

## Key Features

- **Parallel Image Processing**: Each pixel is processed by a separate CUDA thread, demonstrating massive parallelism
- **Multiple Filter Implementations**: Gaussian blur, Sobel edge detection, sharpening, and emboss filters
- **Performance Benchmarking**: Direct comparison between CPU and GPU implementations
- **Synthetic Test Image Generator**: Creates test images with patterns ideal for filter testing

## Technical Details

### CUDA Implementation

The project uses CUDA to accelerate image processing operations. Each filter is implemented both as a sequential CPU version and a parallel GPU version. The implementation demonstrates several important CUDA concepts:

1. **Thread Organization**: Uses 2D grid and block structures to match the 2D nature of image data
2. **Memory Management**: Efficient transfers between host and device memory
3. **Kernel Optimization**: Carefully designed kernels to maximize throughput
4. **Error Handling**: Robust CUDA error checking

### Filter Implementations

1. **Gaussian Blur**: 
   - Implements a separable Gaussian convolution for efficient blurring
   - Dynamically generates blur kernels with configurable radius
   
2. **Sobel Edge Detection**:
   - Implements horizontal and vertical gradient computation
   - Combines gradients to compute edge magnitude
   
3. **Sharpening Filter**:
   - Uses a 3x3 kernel to enhance edges while preserving details
   
4. **Emboss Filter**:
   - Creates a 3D-like effect by emphasizing directional transitions

## Performance

The project includes a benchmarking mode that measures performance across all filter implementations. Typical performance improvements of GPU over CPU implementations:

- Gaussian Blur: 10-30x speedup
- Sobel Edge Detection: 15-40x speedup
- Sharpening: 15-35x speedup
- Emboss: 15-35x speedup

Actual performance varies based on image size and GPU capabilities.

## Building and Running

See [Installation Guide](INSTALL.md) for detailed instructions on setting up and running the project.

## Example Usage

```bash
# Generate a test image
./generate_test_image test_image.jpg 1920 1080

# Run Gaussian blur filter
./gpu_image_processor test_image.jpg gaussian blurred_image.jpg

# Run benchmark for all filters
./gpu_image_processor test_image.jpg benchmark
```

## Requirements

- CUDA Toolkit (11.0 or later)
- OpenCV (4.0 or later)
- C++ compiler with C++11 support
- CMake (3.10 or later)

## Learning Outcomes

This project demonstrates several important concepts in GPU computing:

1. How to structure problems for parallel computation
2. Effective memory management between host and device
3. Performance comparison methodologies between CPU and GPU implementations
4. CUDA kernel design and optimization techniques 