cmake_minimum_required(VERSION 3.10)
project(cev_offtrack_planner_ros2 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)

# CEV
find_package(cev_msgs REQUIRED)

# Add subdirectory for the library
add_subdirectory(cev_planner)

# Include directories for project headers
include_directories(
    include
    cev_planner/include
)

# Source files in the main src folder
set(SOURCES
    src/planner_node.cpp
)

add_executable(planner_node ${SOURCES})

ament_target_dependencies(planner_node
    rclcpp
    cev_msgs
    nav_msgs
    Eigen3
    tf2
    tf2_ros
    tf2_geometry_msgs
    tf2_eigen
)

target_link_libraries(planner_node
    cev_planner
)

# Declare the executable
install(TARGETS planner_node
    DESTINATION lib/${PROJECT_NAME})

# Install additional resources
install(DIRECTORY launch config
    DESTINATION share/${PROJECT_NAME})

# Ament package configuration
ament_package()