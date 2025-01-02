# src/CMakeLists.txt

add_executable(GraphicsProject main.cpp graphics/Graphics.cpp)
target_include_directories(GraphicsProject PRIVATE ../include)

# Link external libraries if needed
target_link_libraries(GraphicsProject tinyobjloader polyscope Eigen3::Eigen)
