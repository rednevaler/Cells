cmake_minimum_required(VERSION 3.9)
project(full_project)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_executable(full_project main.cpp contrast.cpp alignment.cpp laplacianZeroCrossings.cpp redraw.cpp project.h)

target_link_libraries(full_project ${OpenCV_LIBS})