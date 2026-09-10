cmake_minimum_required(VERSION 3.8)
# rosidl 生成的是 C 库：必须启用 C，只写 CXX 会让 CMake 定不出链接语言
project(prm_interfaces LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ReplanPlan.srv"
  DEPENDENCIES geometry_msgs nav_msgs sensor_msgs
)

ament_package()
