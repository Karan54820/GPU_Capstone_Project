#include <iostream>
#include <opencv2/opencv.hpp>

// Generate a test image with patterns useful for testing image processing algorithms
void generateTestImage(const std::string& filename, int width = 800, int height = 600) {
    // Create a black image
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Draw some geometric shapes
    
    // Gradient from black to white
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width / 4; x++) {
            int intensity = 255 * x / (width / 4);
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
        }
    }
    
    // Colored squares
    int squareSize = 100;
    cv::rectangle(image, cv::Point(width / 4, 0), cv::Point(width / 4 + squareSize, squareSize), cv::Scalar(255, 0, 0), -1);
    cv::rectangle(image, cv::Point(width / 4 + squareSize, 0), cv::Point(width / 4 + 2 * squareSize, squareSize), cv::Scalar(0, 255, 0), -1);
    cv::rectangle(image, cv::Point(width / 4 + 2 * squareSize, 0), cv::Point(width / 4 + 3 * squareSize, squareSize), cv::Scalar(0, 0, 255), -1);
    
    // Draw a grid
    for (int y = squareSize; y < 2 * squareSize; y++) {
        for (int x = width / 4; x < width / 4 + 3 * squareSize; x++) {
            if ((x / 10) % 2 == (y / 10) % 2) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
        }
    }
    
    // Draw circles with different radii
    cv::circle(image, cv::Point(width / 2, height / 2), 50, cv::Scalar(255, 0, 0), -1);
    cv::circle(image, cv::Point(width / 2, height / 2), 100, cv::Scalar(0, 255, 0), 3);
    cv::circle(image, cv::Point(width / 2, height / 2), 150, cv::Scalar(0, 0, 255), 5);
    
    // Draw some lines for edge detection
    cv::line(image, cv::Point(0, 3 * height / 4), cv::Point(width, 3 * height / 4), cv::Scalar(255, 255, 255), 2);
    cv::line(image, cv::Point(width / 3, 2 * height / 3), cv::Point(2 * width / 3, 3 * height / 4), cv::Scalar(255, 255, 255), 2);
    cv::line(image, cv::Point(2 * width / 3, 2 * height / 3), cv::Point(width / 3, 3 * height / 4), cv::Scalar(255, 255, 255), 2);
    
    // Add some text
    cv::putText(image, "GPU Image Processing", cv::Point(50, height - 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    
    // Save the image
    bool success = cv::imwrite(filename, image);
    
    if (success) {
        std::cout << "Test image generated and saved to: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save test image to: " << filename << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string filename = argc > 1 ? argv[1] : "sample_images/test_image.jpg";
    int width = argc > 2 ? std::stoi(argv[2]) : 800;
    int height = argc > 3 ? std::stoi(argv[3]) : 600;
    
    generateTestImage(filename, width, height);
    
    return 0;
} 