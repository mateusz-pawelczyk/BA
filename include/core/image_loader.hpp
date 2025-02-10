#pragma once

#include "core/data_loader.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

class ImageLoader : public DataLoader
{
public:
    ImageLoader() = default;
    ~ImageLoader() = default;

    void load_data(const std::string &path) override;
    void load_data(const Eigen::MatrixXd &data) override;

    void load_image(const std::string &path);

    void visualize();

private:
    std::vector<cv::Mat> images;
};