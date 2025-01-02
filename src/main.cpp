#include <iostream>
#include <Eigen/Core>
#include <polyscope/polyscope.h>

int main() {
    std::cout << "Initializing Polyscope..." << std::endl;

    polyscope::init();
    polyscope::show();

    Eigen::Vector3f vec(1.0f, 2.0f, 3.0f);
    std::cout << "Eigen Vector: " << vec.transpose() << std::endl;

    return 0;
}
