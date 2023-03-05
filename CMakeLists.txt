cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_COMPILER "/opt/rh/devtoolset-8/root/usr/bin/g++")
set(OpenCV_DIR "../opencv-3.3.0/build/unix-install/")
project(myproject)
# Find OpenCV package (problem here)
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the source files
set(SOURCES main.cpp)

# Create the executable
add_executable(myproject ${SOURCES})

# Link against OpenCV libraries
target_link_libraries(myproject ${OpenCV_LIBS})

# Create a custom target to clean the build artifacts (didn't work)
#add_custom_target(clean
#    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/build
#    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/build
#    COMMENT "Cleaned build artifacts"
#)
# run with "cmake --build . --target clean"