#
# CMakeLists.txt (Root)
#
# Configures the overall build for GraphicsProject.
#

# ------------------------------------------------------------------------------
# CMake Minimum Version
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14)

# ------------------------------------------------------------------------------
# Project Name and Settings
# ------------------------------------------------------------------------------
project(GraphicsProject LANGUAGES CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Use pure ISO C++ (no compiler-specific extensions)

# Export compile commands for IDEs, LSPs, and code analysis tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")
# set(CMAKE_BUILD_TYPE Release)

# ------------------------------------------------------------------------------
# Prevent In-Source Builds (Forces Build Directory)
# ------------------------------------------------------------------------------
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Please create a 'build' directory and build from there.")
endif()

# ------------------------------------------------------------------------------
# Dependencies and External Libraries
# ------------------------------------------------------------------------------
# External dependencies are managed separately in Dependencies.cmake
include(cmake/Dependencies.cmake)

# ------------------------------------------------------------------------------
# Main Library (Core Project Logic)
# ------------------------------------------------------------------------------
# Collect all source files, excluding main.cpp
file(GLOB_RECURSE PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*src/main.cpp$")


# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

add_library(GraphicsProjectLib ${PROJECT_SOURCES})

# Include directories for the library
target_include_directories(GraphicsProjectLib
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/polyscope/include"  # Polyscope includes
    "${EIGEN_INCLUDE_DIR}"  # Eigen includes
    "${OpenCV_INCLUDE_DIRS}"  # OpenCV includes
)

# Link external dependencies
target_link_libraries(GraphicsProjectLib
    PRIVATE
    polyscope
    ${OpenCV_LIBS}
    Ceres::ceres
)

# ------------------------------------------------------------------------------
# Main Executable
# ------------------------------------------------------------------------------
# Define the main executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link libraries to the main executable
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    GraphicsProjectLib
    polyscope
    Ceres::ceres
)

# ------------------------------------------------------------------------------
# Testing Executable
# ------------------------------------------------------------------------------
# Collect all test files
file(GLOB_RECURSE PROJECT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

# Define the test executable
add_executable(${PROJECT_NAME}_tests ${PROJECT_TEST_SOURCES})

# Include directories for the test executable
target_include_directories(${PROJECT_NAME}_tests
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/polyscope/include"
    "${EIGEN_INCLUDE_DIR}"
)

# Link libraries to the test executable
target_link_libraries(${PROJECT_NAME}_tests
    PRIVATE
    GraphicsProjectLib  # Link core project library
    gtest
    gtest_main
)

# Automatically discover and register tests
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_tests)



# ------------------------------------------------------------------------------
# End of File
# ------------------------------------------------------------------------------
