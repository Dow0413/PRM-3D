cmake_minimum_required(VERSION 3.1...3.20)
project(PRM-MultiFloor)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.0 REQUIRED)
message(STATUS "OpenCV is set to ${OpenCV_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/lib
    ${PROJECT_SOURCE_DIR}/lib/PRMMAP
    ${PROJECT_SOURCE_DIR}/lib/CSubdivision
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples)

# PRM 核心库
add_library(PRMMAP STATIC
    lib/PRMMAP/PRMmapcd.cpp
    lib/PRMMAP/PRMcommon.cpp
    lib/PRMMAP/PRMthpp.cpp
    lib/PRMMAP/PRMmultifloor.cpp
    lib/PRMMAP/debug.cpp
)
target_include_directories(PRMMAP PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PRMMAP PUBLIC ${OpenCV_LIBS})

# 跨楼层路径规划测试程序
add_executable(testMultiFloor
    test/testMultiFloor.cpp
)
target_link_libraries(testMultiFloor PUBLIC
    ${OpenCV_LIBS}
    PRMMAP
)

# 2D 交互仿真：规划 + 避障 + 反应式重规划
add_executable(sim2d
    test/sim2d.cpp
)
target_link_libraries(sim2d PUBLIC
    ${OpenCV_LIBS}
    PRMMAP
)
