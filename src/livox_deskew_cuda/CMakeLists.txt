cmake_minimum_required(VERSION 3.18)
project(livox_deskew_cuda LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# NVIDIA AGX Orin SM_87 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 87)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(DEPENDENCIES
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_msgs
  tf2_geometry_msgs
  builtin_interfaces
)

# CUDA library
add_library(cuda_deskew_lib STATIC
  cuda/cuda_deskew.cu
)

target_include_directories(cuda_deskew_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda>
  $<INSTALL_INTERFACE:include>
)

set_target_properties(cuda_deskew_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(cuda_deskew_lib
  CUDA::cudart
)

# C++ ROS2 library
add_library(${PROJECT_NAME}_lib SHARED
  src/pose_buffer.cpp
  src/deskew_node.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}_lib
  cuda_deskew_lib
  CUDA::cudart
)

ament_target_dependencies(${PROJECT_NAME}_lib ${DEPENDENCIES})

# Executable
add_executable(${PROJECT_NAME}_node src/main.cpp)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_lib
  cuda_deskew_lib
)

ament_target_dependencies(${PROJECT_NAME}_node ${DEPENDENCIES})

# Install
install(TARGETS cuda_deskew_lib ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY cuda/
  DESTINATION include/${PROJECT_NAME}/cuda
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_cuda_deskew test/test_cuda_deskew.cpp)
  target_link_libraries(test_cuda_deskew ${PROJECT_NAME}_lib cuda_deskew_lib)
  ament_target_dependencies(test_cuda_deskew ${DEPENDENCIES})
endif()

ament_package()
