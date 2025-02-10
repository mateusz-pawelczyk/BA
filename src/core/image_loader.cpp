#include "core/image_loader.hpp"
#include <iostream>

void ImageLoader::load_image(const std::string &path)
{
    // Read the image in grayscale mode.cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        throw std::runtime_error("Error: Could not open or find the image at " + path);
    }

    // Convert the image to double precision and scale pixel values to [0.0, 1.0].
    cv::Mat img_double;
    img.convertTo(img_double, CV_64F, 1.0 / 255.0);

    // Ensure that the image data is stored continuously in memory.
    if (!img_double.isContinuous())
    {
        img_double = img_double.clone();
    }

    images.push_back(img_double);

    // Get the total number of pixels.
    const int totalPixels = static_cast<int>(img_double.total());

    // Create an Eigen vector to hold the flattened image.
    Eigen::VectorXd grayscaleVector(totalPixels);

    // Copy the image data into the Eigen vector.
    // Since img_double is continuous and stored in row-major order, we can use memcpy.
    std::memcpy(grayscaleVector.data(), img_double.ptr(), totalPixels * sizeof(double));
}

void ImageLoader::load_data(const Eigen::MatrixXd &data)
{
}

cv::Mat createImageGrid(const std::vector<cv::Mat> &images)
{
    if (images.empty())
    {
        std::cerr << "No images to display!" << std::endl;
        return cv::Mat();
    }

    // Determine grid size (rows and columns) that is as square as possible.
    int numImages = static_cast<int>(images.size());
    int gridCols = static_cast<int>(std::ceil(std::sqrt(numImages)));
    int gridRows = static_cast<int>(std::ceil(static_cast<double>(numImages) / gridCols));

    // Assume all images are the same size.
    int imgHeight = images[0].rows;
    int imgWidth = images[0].cols;
    int type = images[0].type();

    // Create a composite image with black background.
    cv::Mat gridImage = cv::Mat::zeros(gridRows * imgHeight, gridCols * imgWidth, type);

    // Copy each image into its corresponding grid cell.
    for (int i = 0; i < numImages; ++i)
    {
        // Determine the grid cell coordinates.
        int rowIdx = i / gridCols;
        int colIdx = i % gridCols;
        // Define the region of interest (ROI) in the composite image.
        cv::Rect roi(colIdx * imgWidth, rowIdx * imgHeight, imgWidth, imgHeight);
        // Make sure the ROI is within gridImage bounds, then copy.
        if (roi.x + roi.width <= gridImage.cols && roi.y + roi.height <= gridImage.rows)
        {
            images[i].copyTo(gridImage(roi));
        }
    }
    return gridImage;
}

void ImageLoader::visualize()
{
    cv::Mat gridImage = createImageGrid(images);
    if (gridImage.empty())
    {
        std::cerr << "Error: Failed to create grid image." << std::endl;
        return;
    }

    cv::imshow("Images", gridImage);

    cv::waitKey(0);
}