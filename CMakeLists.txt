cmake_minimum_required(VERSION 3.10)
project(gpu_image_processor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

# Find OpenCV package
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION} at ${OpenCV_INCLUDE_DIRS}")

# Include directories
include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_50)

# Add executable for image processor
file(GLOB SOURCES "src/*.cpp" "src/*.cu")
list(FILTER SOURCES EXCLUDE REGEX "src/generate_test_image.cpp")
cuda_add_executable(gpu_image_processor ${SOURCES})

# Link libraries for image processor
target_link_libraries(gpu_image_processor ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Add executable for test image generator
add_executable(generate_test_image src/generate_test_image.cpp)
target_link_libraries(generate_test_image ${OpenCV_LIBS})

# Create directories for sample images and results
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/images)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/results)

# Copy sample images to build directory
file(GLOB SAMPLE_IMAGES "sample_images/*")
foreach(SAMPLE_IMAGE ${SAMPLE_IMAGES})
    file(COPY ${SAMPLE_IMAGE} DESTINATION ${CMAKE_BINARY_DIR}/images)
endforeach() 