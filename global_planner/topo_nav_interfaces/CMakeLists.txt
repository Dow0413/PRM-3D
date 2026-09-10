cmake_minimum_required(VERSION 3.8)
project(topo_nav_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Generate message and service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TopoNode.msg"
  "msg/TopoSegment.msg"
  "msg/TopoPathMeta.msg"
  "srv/FollowPath.srv"
  "srv/QueryPath.srv"
  "srv/QueryNearestTopo.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
