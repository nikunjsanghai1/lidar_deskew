cmake_minimum_required(VERSION 3.8)
project(livox_deskew)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
find_package(Eigen3 REQUIRED)

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

# Library
add_library(${PROJECT_NAME}_lib SHARED
  src/pose_buffer.cpp
  src/point_time_extractor.cpp
  src/deskew_node.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(${PROJECT_NAME}_lib ${DEPENDENCIES})

# Executable
add_executable(${PROJECT_NAME}_node src/main.cpp)

target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}_lib)

ament_target_dependencies(${PROJECT_NAME}_node ${DEPENDENCIES})

# Install
install(TARGETS ${PROJECT_NAME}_lib
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

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_pose_buffer test/test_pose_buffer.cpp)
  target_link_libraries(test_pose_buffer ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_pose_buffer ${DEPENDENCIES})

  ament_add_gtest(test_deskew_stationary test/test_deskew_stationary.cpp)
  target_link_libraries(test_deskew_stationary ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_deskew_stationary ${DEPENDENCIES})
endif()

ament_package()
