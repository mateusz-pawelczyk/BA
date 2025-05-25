#
# CMakeLists.txt (Root)
#
# Configures the overall build for GraphicsProject.
#

# ------------------------------------------------------------------------------
# CMake Minimum Version
# ------------------------------------------------------------------------------
# Specifies the minimum version of CMake required to build this project.
# Updating this version might be necessary if you start using newer CMake features.
cmake_minimum_required(VERSION 3.14)

# ------------------------------------------------------------------------------
# Project Definition
# ------------------------------------------------------------------------------
# Defines the project name and the primary language (C++).
# This also sets up default variables like PROJECT_NAME, PROJECT_SOURCE_DIR, etc.
project(GraphicsProject LANGUAGES CXX)

# ------------------------------------------------------------------------------
# C++ Standard Configuration
# ------------------------------------------------------------------------------
# Sets the C++ standard to C++20.
# CMAKE_CXX_STANDARD_REQUIRED ON: Enforces the standard; build will fail if not supported.
# CMAKE_CXX_EXTENSIONS OFF: Disables compiler-specific extensions (e.g., -gnu++20)
#                           favoring pure ISO C++ (e.g., -std=c++20).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Build Configuration
# ------------------------------------------------------------------------------
# Export compile commands for IDEs, LSPs, and code analysis tools (e.g., Clangd).
# This generates a 'compile_commands.json' file in the build directory.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prevent in-source builds to keep the source directory clean.
# Users should create a separate build directory (e.g., 'build/').
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Please create a 'build' directory (e.g., 'mkdir build && cd build') and run CMake from there.")
endif()

# ------------------------------------------------------------------------------
# Project Structure & Output Directories (Optional)
# ------------------------------------------------------------------------------
# You can customize output directories for executables, libraries, and archives.
# Example:
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ------------------------------------------------------------------------------
# Dependencies Management
# ------------------------------------------------------------------------------
# External dependencies and their setup are managed in a separate file.
# This keeps the root CMakeLists.txt focused on the project's own targets.
include(cmake/Dependencies.cmake)
# Future: If you have project-specific CMake modules or functions, you might add:
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# ------------------------------------------------------------------------------
# Main Library (GraphicsProjectLib)
# ------------------------------------------------------------------------------
# This library encapsulates the core logic of your project.

# Collect all source files for the library from the 'src' directory,
# excluding main.cpp and old_main.cpp which are entry points for executables.
# Note: While file(GLOB_RECURSE) is convenient, explicitly listing source files
# ensures CMake reliably detects new files or removals.
file(GLOB_RECURSE PROJECT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*src/main\\.cpp$")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*src/old_main\\.cpp$")
# Future: If you add more specific executables with their own main files in src/,
# add them to the EXCLUDE REGEX list above.

# Define the static library target.
add_library(GraphicsProjectLib STATIC ${PROJECT_SOURCES})

# Set properties for GraphicsProjectLib.
target_include_directories(GraphicsProjectLib
    PUBLIC
        # Include directory for the library's own public headers.
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        # Eigen include directory (made available via the Eigen3::Eigen target).
        # If Eigen types are used in public headers of GraphicsProjectLib,
        # linking Eigen3::Eigen PUBLICLY is preferred.
        # If Eigen3::Eigen target is available and correctly configured,
        # this manual include might be redundant if linked PUBLICLY.
        # For clarity, if Eigen3::Eigen is an INTERFACE library target that
        # correctly sets its INTERFACE_INCLUDE_DIRECTORIES, just linking it
        # PUBLIC is enough.
        # See Dependencies.cmake for EIGEN_INCLUDE_DIR setup. We'll link Eigen3::Eigen.
    PRIVATE
        # Include directories for internal implementation details, if any.
)

# Link dependencies to GraphicsProjectLib.
# PUBLIC: If Polyscope or Ceres types/headers are used in GraphicsProjectLib's
#         public headers (in your 'include' directory). This makes them
#         available to anything that links against GraphicsProjectLib.
# PRIVATE: If Polyscope or Ceres are only used in GraphicsProjectLib's .cpp files
#          (implementation detail).
target_link_libraries(GraphicsProjectLib
    PUBLIC
        Eigen3::Eigen         # Eigen is typically used in headers.
        polyscope             # Assuming Polyscope might be in public API.
        Ceres::ceres          # Assuming Ceres might be in public API.
                              # If not, change to PRIVATE.
    # PRIVATE
        # Add other private dependencies here.
)

# OpenMP support for the library (if sources use OpenMP).
# These flags are for compiling the library's sources.
if(OpenMP_CXX_FOUND)
    target_compile_options(GraphicsProjectLib PRIVATE ${OpenMP_CXX_FLAGS})
    # For a STATIC library, compile flags are usually sufficient. The final linking
    # of the OpenMP runtime happens with the executable.
    # If GraphicsProjectLib's public headers required OpenMP headers/constructs,
    # you might consider linking OpenMP::OpenMP_CXX PUBLICLY to propagate requirements.
    # However, this is less common for static libraries where OpenMP usage is internal.
endif()

# ------------------------------------------------------------------------------
# Main Executable (GraphicsProject)
# ------------------------------------------------------------------------------
# Defines the main executable for the project.

add_executable(${PROJECT_NAME} src/main.cpp)

# Link GraphicsProjectLib to the executable.
# PRIVATE is appropriate here as the executable is the final product.
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        GraphicsProjectLib
        # Transitive dependencies from GraphicsProjectLib (like Eigen, Polyscope, Ceres)
        # will be linked automatically if they were PUBLIC on GraphicsProjectLib.
        # If they were PRIVATE to GraphicsProjectLib, and your main.cpp also directly uses them,
        # you would need to link them here again:
        # polyscope
        # Ceres::ceres
)

# OpenMP support for the executable.
# This ensures the executable links against the OpenMP runtime.
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# ------------------------------------------------------------------------------
# Testing (Placeholder)
# ------------------------------------------------------------------------------
# Future: If you add tests (e.g., using CTest with GoogleTest or Catch2):
# 1. Enable testing: enable_testing() (typically at the top or in a dedicated section)
# 2. Add GTest/Catch2 via cmake/Dependencies.cmake or FetchContent.
# 3. Define test executables: add_executable(MyTests test/my_tests.cpp)
# 4. Link test executables against your library and the test framework.
# 5. Register tests: add_test(NAME MyTestCommand COMMAND MyTests)

# Example for future testing setup:
# if(BUILD_TESTING) # Add an option(BUILD_TESTING "Build tests" ON)
#   enable_testing()
#   add_subdirectory(tests) # Assuming tests have their own CMakeLists.txt
# endif()

# ------------------------------------------------------------------------------
# Installation (Placeholder)
# ------------------------------------------------------------------------------
# Future: If you want to install your project:
# install(TARGETS ${PROJECT_NAME} GraphicsProjectLib
#         RUNTIME DESTINATION bin
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib/static)
# install(FILES include/GraphicsProject/MyHeader.h DESTINATION include/GraphicsProject)
# See CMake documentation for 'install' command for more details.

# ------------------------------------------------------------------------------
# End of Root CMakeLists.txt
# ------------------------------------------------------------------------------
message(STATUS "✔ Root CMakeLists.txt processing complete.")