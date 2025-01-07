#
# Dependencies.cmake
# 
# This file handles the integration of external libraries like Polyscope and Eigen.
# All configurations, options, and subdirectory additions for external libs go here,
# keeping the root CMakeLists.txt cleaner and more maintainable.
#

#############################
#        POLYSCOPE
#############################
# Turn off any features we don't need for Polyscope (examples, tests, etc.).
# We only want the Polyscope library for our project.
option(POLYSCOPE_BUILD_SHARED_LIBS  "Build Polyscope as a shared library" OFF)
option(POLYSCOPE_BUILD_EXAMPLES     "Build Polyscope examples"            OFF)
option(POLYSCOPE_IMGUI_ENABLED      "Enable ImGui within Polyscope"       ON) 
# ^ Typically ON for Polyscope's GUI, but can be toggled as needed.

# Add the Polyscope directory
# This will pick up Polyscope's own CMakeLists.txt file under external/polyscope/
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/polyscope")

#############################
#          EIGEN
#############################
# Eigen is header-only, so we do not build anything for it.
# We only need to set the include path to point to the Eigen folder.

set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen")
# In case you want to confirm if a CMakeLists.txt exists in Eigen, you can
# add_subdirectory(...) it if you prefer. Usually, for Eigen, you only need the path.

# Optionally, you can wrap Eigen usage in a library target for clarity:
# add_library(eigen INTERFACE)
# target_include_directories(eigen INTERFACE "${EIGEN_INCLUDE_DIR}")

#
# End of Dependencies.cmake
#
