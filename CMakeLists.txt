cmake_minimum_required(VERSION 3.14)
project(GraphicsProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(GraphicsProject src/main.cpp)
target_include_directories(GraphicsProject PRIVATE include)
