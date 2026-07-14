## 📄 中文 README

# PRM-跨楼层路径规划 (PRM-MultiFloor)

这是一个基于 PRM（Probabilistic Roadmap）的跨楼层路径规划库，支持在多层建筑中进行系留/非系留机器人的最优路径规划。

## 🧩 功能模块

- **PRM-TPP** (Configuration Deformation Tree - Tethered Path Planner)
  系留机器人最优路径规划
  实现于：`BImap::THPPoptimalPlanner`

- **PRM-UTPP** (Configuration Deformation Tree - Untethered Path Planner)
  非系留机器人最优路径规划
  实现于：`BImap::UTHPPoptimalPlanner`

- **PRM-MultiFloor** (跨楼层路径规划)
  支持多层建筑的跨楼层路径规划
  实现于：`PRMMultiFloor` 类

## ⚙️ 运行要求

### C++ 编译依赖
- OpenCV 4.0 或更高版本
- C++14 兼容编译器
- nlohmann/json 库

## 📁 目录结构

```
PRM-3D/
├── lib/PRMMAP/
│   ├── PRMmap.h          # 核心头文件
│   ├── PRMmapcd.cpp      # 地图构建
│   ├── PRMcommon.cpp     # 通用函数
│   ├── PRMthpp.cpp       # THPP 路径规划
│   ├── PRMmultifloor.h   # 跨楼层头文件
│   ├── PRMmultifloor.cpp # 跨楼层规划实现
│   └── debug.cpp         # 调试函数
├── test/
│   └── testMultiFloor.cpp # 跨楼层规划测试
├── CMakeLists.txt
└── MULTIFLOOR_README.md   # 详细使用文档
```

## 🚀 快速开始

### 1. 编译

```bash
mkdir build && cd build
cmake ..
make -j4
```

### 2. 运行测试

```bash
../Examples/testMultiFloor
```

### 3. 基本用法

```cpp
#include "PRMmultifloor.h"

// 创建规划器
PRMMultiFloor planner;
planner.initialize(2);  // 2 层楼

// 加载楼层地图
planner.loadFloorGraph(0, floor1Graph);
planner.loadFloorGraph(1, floor2Graph);

// 添加楼梯间连接
planner.addFloorConnection(0, 1, stairs_f1, stairs_f2, FloorConnectionType::STAIRS);

// 创建任务
MultiFloorTHPPtask task;
task.Origin = {100, 100};
task.OriginFloor = 0;
task.Goal = {200, 200};
task.GoalFloor = 1;
task.TetherLength = 50.0;

// 执行规划
double cost = planner.multiFloorPlanner(task);
```

## 📖 详细文档

完整的使用说明和 API 文档请参阅 [MULTIFLOOR_README.md](MULTIFLOOR_README.md)

## 💡 注意事项

- 确保所有楼层地图使用相同的坐标系统
- 连接点（楼梯间）必须位于自由空间内
- 系留模式下，路径总长度不能超过 TetherLength

## 🔧 故障排除

### 编译错误
- 确认 OpenCV 版本 >= 4.0
- 确认 nlohmann/json 库已正确安装

### 路径规划失败
- 检查起始点/目标点是否在自由空间内
- 确认楼层连接是否正确添加
- 检查系留长度是否足够
