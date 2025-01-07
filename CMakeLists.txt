#
# CMakeLists.txt (Root)
#
# This file configures the overall build for GraphicsProject.
# It defines project settings, includes external dependencies, and sets up the main executable.
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
add_compile_options(
    -Wall         # Show all standard warnings
    -Wextra       # Enable extra warnings
    -Wpedantic    # Enforce strict ISO C++ compliance
)

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

# ------------------------------------------------------------------------------
# Main Executable
# ------------------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}  # Headers can be included for IDE indexing, not needed for compilation
)

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
# Add project include directory
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Ensure Eigen's header-only include path is available
target_include_directories(${PROJECT_NAME}
    PRIVATE 
    "${EIGEN_INCLUDE_DIR}"
)

# ------------------------------------------------------------------------------
# Linking External Libraries
# ------------------------------------------------------------------------------
# Link Polyscope (Ensure Polyscope is correctly added via Dependencies.cmake)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    polyscope
)

# ------------------------------------------------------------------------------
# Post-Build Actions (Optional)
# ------------------------------------------------------------------------------
# This section can handle asset copying, shader distribution, or other post-build tasks.
# For example:
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#     "${CMAKE_SOURCE_DIR}/assets"
#     "${CMAKE_BINARY_DIR}/assets"
# )

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
# This CMakeLists.txt file is structured to ensure clear separation of project components
# and maintainability, while enforcing good practices for dependency management and warnings.
