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

# Set C++ standard to C++20 (compatible with GCC 11.4.0)
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
# Compiler Warnings (For Clean Code)
# ------------------------------------------------------------------------------
# add_compile_options(
#     -Wall         # Show all standard warnings
# )

# ------------------------------------------------------------------------------
# Dependencies and External Libraries
# ------------------------------------------------------------------------------
# External dependencies are managed separately in Dependencies.cmake
include(cmake/Dependencies.cmake)

# ------------------------------------------------------------------------------
# Source and Header File Management
# ------------------------------------------------------------------------------
# Automatically collect all .cpp and .h/.hpp/.tpp files for compilation
file(GLOB_RECURSE PROJECT_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
file(GLOB_RECURSE PROJECT_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.tpp"
)

file(GLOB_RECURSE PROJECT_TESTS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

# ------------------------------------------------------------------------------
# Main Library (Shared Project Logic)
# ------------------------------------------------------------------------------
add_library(GraphicsProjectLib
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}  # Headers included for IDE indexing
)

# Ensure all required include directories are added to the library
target_include_directories(GraphicsProjectLib
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/polyscope/include"  # Add Polyscope includes
    "${EIGEN_INCLUDE_DIR}"  # Add Eigen includes
)

target_link_libraries(GraphicsProjectLib
    PRIVATE
    polyscope  # Link Polyscope
)

# ------------------------------------------------------------------------------
# Main Executable
# ------------------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    src/main.cpp  # Only include the main file here
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    GraphicsProjectLib
    polyscope  # Link Polyscope
)

# ------------------------------------------------------------------------------
# Testing Executable
# ------------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_tests
    ${PROJECT_TESTS}
)

target_include_directories(${PROJECT_NAME}_tests
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/polyscope/include"  # Add Polyscope includes
    "${EIGEN_INCLUDE_DIR}"  # Add Eigen includes
)

target_link_libraries(${PROJECT_NAME}_tests
    PRIVATE
    GraphicsProjectLib  # Link shared project logic
    gtest
    gtest_main
    polyscope  # Link Polyscope
)

# Automatically discover and register tests
gtest_discover_tests(${PROJECT_NAME}_tests)

# ------------------------------------------------------------------------------
# End of File
# ------------------------------------------------------------------------------
