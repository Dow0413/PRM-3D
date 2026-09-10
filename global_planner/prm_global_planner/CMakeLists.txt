cmake_minimum_required(VERSION 3.8)
project(prm_global_planner LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(prm_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

# --- PRM-3D core (vendored from the PRM-3D repo lib/PRMMAP) ---
add_library(prm_core STATIC
  src/prm_core/PRMmapcd.cpp
  src/prm_core/PRMcommon.cpp
  src/prm_core/PRMthpp.cpp
  src/prm_core/PRMmultifloor.cpp
  src/prm_core/debug.cpp
)
target_include_directories(prm_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/prm_core>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(prm_core PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

# --- ROS 2 node: the "prm" global path source ---
add_executable(prm_planner_node src/prm_planner_node.cpp)
target_link_libraries(prm_planner_node prm_core)
ament_target_dependencies(prm_planner_node
  rclcpp
  prm_interfaces
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2_ros
  tf2_sensor_msgs
)

install(TARGETS prm_core prm_planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
