# external/CMakeLists.txt

include(FetchContent)

# tinyobjloader
FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG v1.0.6
)
FetchContent_MakeAvailable(tinyobjloader)

# Polyscope
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY git@github.com:nmwsharp/polyscope.git
  GIT_TAG v2.3.0
)
FetchContent_MakeAvailable(polyscope)


# Eigen (Header-only)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Link libraries globally (optional, depending on usage)
add_library(tinyobjloader ALIAS tinyobjloader)
add_library(polyscope ALIAS polyscope)
