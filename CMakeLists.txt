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
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*src/old_main.cpp$")

# Find required package: Ceres
find_package(Ceres REQUIRED)

add_library(GraphicsProjectLib ${PROJECT_SOURCES})

# Include directories for the library
target_include_directories(GraphicsProjectLib
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/polyscope/include"  # Polyscope includes
    "${EIGEN_INCLUDE_DIR}"  # Eigen includes
)

# Link external dependencies
target_link_libraries(GraphicsProjectLib
    PRIVATE
    polyscope
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
# End of File
# ------------------------------------------------------------------------------
