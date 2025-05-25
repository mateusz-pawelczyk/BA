#
# Dependencies.cmake
#
# This file handles the configuration and integration of all external third-party
# libraries used by the GraphicsProject.
#

message(STATUS "Processing external dependencies...")

# ------------------------------------------------------------------------------
# Eigen (Header-only linear algebra library)
# ------------------------------------------------------------------------------
# Eigen is typically header-only. We need to tell CMake where to find its headers.
# Option 1: Using a bundled version (as in your original setup)
set(EIGEN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen" CACHE PATH "Path to Eigen headers")
# Ensure the directory exists
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}")
    message(WARNING "Eigen include directory not found at ${EIGEN_INCLUDE_DIR}. Please check the path.")
else()
    # Create an INTERFACE library target for Eigen. This is a modern CMake way
    # to handle header-only libraries. It allows us to use Eigen3::Eigen
    # as a dependency in target_link_libraries, which is cleaner.
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE "${EIGEN_INCLUDE_DIR}")
    message(STATUS "✔ Configured Eigen3::Eigen target using bundled version from: ${EIGEN_INCLUDE_DIR}")
endif()

# Option 2: Using find_package (if Eigen is installed system-wide or via CMAKE_PREFIX_PATH)
# find_package(Eigen3 REQUIRED) # This would look for Eigen3Config.cmake or eigen3-config.cmake
# if(Eigen3_FOUND)
#    message(STATUS "✔ Found Eigen3 version: ${Eigen3_VERSION_STRING} at ${Eigen3_INCLUDE_DIRS}")
#    # Eigen3::Eigen target is usually created by find_package(Eigen3)
# else()
#    message(FATAL_ERROR "Eigen3 not found. Please install it or set EIGEN_INCLUDE_DIR.")
# endif()
# Choose one of the above methods for Eigen. The bundled approach (Option 1) is kept for consistency
# with your original setup. If Eigen3::Eigen is not found/set by find_package, the add_library above is needed.

# ------------------------------------------------------------------------------
# Ceres Solver (Non-linear optimization library)
# ------------------------------------------------------------------------------
# Configure paths for a locally built/installed Ceres Solver.
# This ensures that find_package(Ceres) picks up your specific version.
set(CERES_INSTALL_DIR "${CMAKE_SOURCE_DIR}/external/ceres-solver/install" CACHE PATH "Path to local Ceres installation")

if(EXISTS "${CERES_INSTALL_DIR}/lib/cmake/Ceres" OR EXISTS "${CERES_INSTALL_DIR}/CeresConfig.cmake")
    # Add the local Ceres installation to CMAKE_PREFIX_PATH so find_package can discover it.
    # list(PREPEND ...) is safer if CMAKE_PREFIX_PATH might be used by other find_package calls.
    list(PREPEND CMAKE_PREFIX_PATH "${CERES_INSTALL_DIR}")
    message(STATUS "ℹ Added local Ceres path to CMAKE_PREFIX_PATH: ${CERES_INSTALL_DIR}")
else()
    message(WARNING "Local Ceres installation not found at ${CERES_INSTALL_DIR}. Will try to find system Ceres or fail if not found and REQUIRED.")
endif()

# Find the Ceres package.
# The Ceres::ceres target will be created, providing include directories and libraries.
find_package(Ceres REQUIRED) # Ceres::ceres target will be available

if(Ceres_FOUND)
    message(STATUS "✔ Found Ceres version: ${CERES_VERSION}")
    message(STATUS "✔ Ceres include directories: ${CERES_INCLUDE_DIRS}") # Provided by Ceres's config
    message(STATUS "✔ Ceres libraries: ${CERES_LIBRARIES}")             # Provided by Ceres's config
else()
    message(FATAL_ERROR "Ceres Solver not found! Please build/install it in 'external/ceres-solver/install' or ensure it's findable by CMake.")
endif()

# ------------------------------------------------------------------------------
# Polyscope (3D visualization library)
# ------------------------------------------------------------------------------
# Polyscope is included as a subdirectory. We can set its CMake options here
# before adding the subdirectory.

# Options for Polyscope build:
option(POLYSCOPE_BUILD_SHARED_LIBS "Build Polyscope as a shared library" OFF)
option(POLYSCOPE_BUILD_EXAMPLES    "Build Polyscope examples" OFF)
option(POLYSCOPE_ENABLE_PROGRAMS   "Build Polyscope standalone programs" OFF) # Often good to turn off for lib-only usage
option(POLYSCOPE_IMGUI_ENABLED     "Enable ImGui within Polyscope" ON) # Typically ON for GUI
option(POLYSCOPE_BACKEND           "Polyscope rendering backend (e.g., openGL_glfw, openGL_SDL2_imgui)" "openGL_glfw") # Example, adjust if needed

# Add the Polyscope subdirectory. This will process its CMakeLists.txt.
# Ensure the path is correct relative to the root project source directory.
if(EXISTS "${CMAKE_SOURCE_DIR}/external/polyscope/CMakeLists.txt")
    add_subdirectory("${CMAKE_SOURCE_DIR}/external/polyscope" "${CMAKE_BINARY_DIR}/polyscope_build") # Optional: build in a separate binary subdir
    message(STATUS "✔ Added Polyscope subdirectory. Target 'polyscope' should be available.")
    # The target 'polyscope' (or similar, check Polyscope's CMake) should export
    # its include directories and library linkage requirements.
else()
    message(FATAL_ERROR "Polyscope CMakeLists.txt not found at ${CMAKE_SOURCE_DIR}/external/polyscope. Please ensure Polyscope is correctly placed.")
endif()

# ------------------------------------------------------------------------------
# OpenMP (Parallelization library)
# ------------------------------------------------------------------------------
# Find OpenMP. This is used for multi-threading capabilities.
# If found, OpenMP_CXX_FOUND will be TRUE, and OpenMP_CXX_FLAGS will be set.
# The target OpenMP::OpenMP_CXX will be available for linking.
find_package(OpenMP) # Not REQUIRED, so the build can proceed without it if desired (adjust logic if strictly needed)
if(OpenMP_CXX_FOUND)
    message(STATUS "✔ Found OpenMP for C++. Flags: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP for C++ not found. Parallel features might be disabled.")
endif()

# ------------------------------------------------------------------------------
# Other Dependencies (Placeholder)
# ------------------------------------------------------------------------------
# Future: Add other dependencies here following a similar pattern:
# 1. Set options for the library (if built as a subdirectory).
# 2. Set hints for find_package (e.g., XXX_ROOT, CMAKE_PREFIX_PATH).
# 3. Use find_package(XXX REQUIRED) or add_subdirectory(external/XXX).
# 4. Ensure the library provides a target (e.g., XXX::xxx or xxx) for linking.
#
# Example:
# find_package(Boost 1.70 REQUIRED COMPONENTS program_options filesystem)
# if(Boost_FOUND)
#   message(STATUS "✔ Found Boost ${Boost_VERSION_STRING}")
#   # Link with Boost::program_options, Boost::filesystem
# endif()

# ------------------------------------------------------------------------------
# Google Test / Testing Frameworks (Removed as per request)
# ------------------------------------------------------------------------------
# If you decide to add tests later:
# 1. Add GoogleTest (e.g., via add_subdirectory or FetchContent).
#    FetchContent_Declare(
#      googletest
#      GIT_REPOSITORY https://github.com/google/googletest.git
#      GIT_TAG    release-1.12.1 # Or a specific commit/tag
#    )
#    FetchContent_MakeAvailable(googletest)
# 2. Link your test executables with GTest::gtest and GTest::gtest_main.
#    Example: target_link_libraries(MyTests PRIVATE GTest::gtest_main GraphicsProjectLib)
# 3. In your root CMakeLists.txt or a dedicated testing CMake file:
#    enable_testing()
#    include(GoogleTest) # If using CTest's GoogleTest module
#    add_test(...)

# ------------------------------------------------------------------------------
message(STATUS "✔ Finished processing external dependencies.")