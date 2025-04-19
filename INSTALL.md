# Installation Guide

This guide will help you set up and run the GPU-Accelerated Image Processing project.

## Prerequisites

To build and run this project, you need the following components:

1. CUDA Toolkit (11.0 or later)
2. OpenCV (4.0 or later)
3. CMake (3.10 or later)
4. A CUDA-capable GPU with compute capability 3.5 or higher
5. C++ compiler with C++11 support (GCC, MSVC, Clang)

## Installing Dependencies

### Windows

#### CUDA Toolkit
1. Download the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Follow the installation instructions provided by NVIDIA
3. Make sure to add CUDA to your PATH environment variable

#### OpenCV
1. Download OpenCV from [the official website](https://opencv.org/releases/)
2. Extract the files to a location of your choice
3. Add the `bin` directory to your PATH environment variable
4. Set an environment variable `OpenCV_DIR` pointing to the build directory containing `OpenCVConfig.cmake`

#### CMake
1. Download CMake from [the official website](https://cmake.org/download/)
2. Install CMake and make sure it's added to your PATH

### Linux

#### CUDA Toolkit
```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

#### OpenCV
```bash
sudo apt update
sudo apt install libopencv-dev
```

#### CMake
```bash
sudo apt update
sudo apt install cmake
```

## Building the Project

### Windows

1. Clone the repository
2. Create a build directory:
```
mkdir build
cd build
```
3. Generate the Visual Studio solution:
```
cmake ..
```
4. Build the project:
```
cmake --build . --config Release
```

### Linux

1. Clone the repository
2. Create a build directory:
```bash
mkdir build
cd build
```
3. Generate the Makefile:
```bash
cmake ..
```
4. Build the project:
```bash
make
```

## Running the Project

### Generating a Test Image

First, generate a test image using the test image generator:

```
./generate_test_image [output_filename] [width] [height]
```

Example:
```
./generate_test_image test_image.jpg 800 600
```

### Running Image Processing

To process an image with the GPU-accelerated filters:

```
./gpu_image_processor [input_image] [filter_type] [output_image]
```

Filter types:
- `gaussian`: Apply Gaussian blur filter
- `sobel`: Apply Sobel edge detection filter
- `sharpen`: Apply sharpening filter
- `emboss`: Apply emboss filter
- `benchmark`: Run all filters and measure performance

Example:
```
./gpu_image_processor test_image.jpg gaussian output_gaussian.jpg
```

## Running Benchmarks

To run performance benchmarks for all filters:

```
./gpu_image_processor test_image.jpg benchmark
```

This will generate results in the `results` directory and print performance statistics comparing CPU and GPU implementations. 