#pragma once

#include <string>
#include <Eigen/Core>

class DataLoader
{
public:
    virtual ~DataLoader() = default;

    virtual void load_data(const std::string &path) = 0;
    virtual void load_data(const Eigen::MatrixXd &data) = 0;

protected:
    Eigen::MatrixXd D;
};