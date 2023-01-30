cmake_minimum_required(VERSION 3.10)
project(myproject CXX)
add_executable(myproject main.cpp)
# Add the path to the FindOpenCV.cmake module to the CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/home/dtrad/opencv-3.3.0/cmake")
find_package(OpenCV REQUIRED)
target_link_libraries(myproject ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})
set(SOURCES matchimage.cpp)
add_executable(myproject ${SOURCES})


