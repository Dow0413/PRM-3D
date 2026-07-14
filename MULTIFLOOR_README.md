# PRM 跨楼层路径规划使用说明

## 概述

这是一个基于 PRM（Probabilistic Roadmap）的**跨楼层路径规划**库，支持**交互式设置楼层连接点**，通过用户点击指定楼梯间位置，实现两层楼之间的最优路径规划。

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake ..
make -j4
```

### 运行测试程序

```bash
../Examples/testMultiFloor
```

### 操作流程

程序启动后会打开两个窗口，分别显示一楼和二楼的 PRM 剖分地图：

| 步骤 | 操作 | 标记颜色 |
|------|------|----------|
| 1 | 在 **Floor 1** 点击设置**楼梯间**位置 | 黄色 |
| 2 | 在 **Floor 2** 点击设置**楼梯间**位置 | 黄色 |
| 3 | 在 **Floor 1** 点击设置**起点** | 绿色 |
| 4 | 在 **Floor 2** 点击设置**终点** | 红色 |

规划完成后，路径将以**白色线条**显示。

## 核心 API

### 1. 初始化规划器

```cpp
#include "PRMmultifloor.h"

PRMMultiFloor planner;
planner.initialize(2);  // 2 层楼
```

### 2. 加载楼层地图

```cpp
BImap floor1Map, floor2Map;
floor1Map.MaptoBInavi("floor1.png", 0.1, 0.9);
floor2Map.MaptoBInavi("floor2.png", 0.1, 0.9);

planner.loadFloorGraph(0, floor1Map.BIgraphList[0]);
planner.loadFloorGraph(1, floor2Map.BIgraphList[1]);
```

### 3. 添加楼层连接

```cpp
// 用户点击获取连接点坐标和多边形索引后
planner.addFloorConnection(0, 1, point1, point2, FloorConnectionType::STAIRS);
planner.setConnectionPolygons(0, polyIndex1, polyIndex2);
```

### 4. 执行路径规划

```cpp
MultiFloorTask task;
task.start = startPoint;
task.startFloor = 0;
task.startPoly = startPolyIndex;
task.goal = goalPoint;
task.goalFloor = 1;
task.goalPoly = goalPolyIndex;

double cost = planner.plan(task);

// 结果在 task.path 中
for (const auto& p : task.path) {
    printf("(%.1f, %.1f)\n", p.x, p.y);
}
```

## 数据结构

### MultiFloorTask

| 字段 | 说明 |
|------|------|
| start | 起点坐标 |
| startFloor | 起点所在楼层 |
| startPoly | 起点所在多边形索引 |
| goal | 终点坐标 |
| goalFloor | 终点所在楼层 |
| goalPoly | 终点所在多边形索引 |
| path | 规划出的路径（点列表） |
| totalCost | 路径总成本 |

### FloorConnection

| 字段 | 说明 |
|------|------|
| id | 连接点唯一 ID |
| type | 连接类型（STAIRS/ELEVATOR/RAMP） |
| floorFrom/floorTo | 连接的起始/目标楼层 |
| pointFrom/pointTo | 连接点坐标 |
| polyIndexFrom/polyIndexTo | 连接点所在多边形索引 |
| cost | 穿越成本（楼层切换代价） |

## 算法原理

### 跨楼层路径规划流程

```
起点 (Floor 1)
    │
    ▼
BFS 搜索多边形路径
    │
    ▼
GetLeastHomotopyPath 计算几何路径
    │
    ▼
到达楼梯间 (Floor 1)
    │
    ▼ 楼层切换 (+15 成本)
    │
    ▼
楼梯间 (Floor 2)
    │
    ▼
BFS 搜索多边形路径
    │
    ▼
GetLeastHomotopyPath 计算几何路径
    │
    ▼
终点 (Floor 2)
```

### PRM 剖分

1. 检测凹顶点
2. 从凹顶点切割自由空间
3. 生成凸多边形剖分
4. 构建多边形邻接图

## 注意事项

### 1. 地图图片要求
- 白色区域 = 自由空间
- 黑色区域 = 障碍物
- `MaptoBInavi` 自动进行多边形拟合和 PRM 剖分

### 2. 坐标系统
- 两层楼使用相同的坐标原点
- 确保地图图片的像素尺寸一致

### 3. 连接点选择
- 必须点击白色区域（自由空间）
- 不能点击黑色障碍物
- 程序自动获取所在多边形索引

## 示例代码

完整示例参见 `test/testMultiFloor.cpp`

## 故障排除

### 编译失败
```
找不到 opencv2/core/core.hpp
```
**解决**：确保 OpenCV 4.0+ 已正确安装

### 路径规划失败
```
error: cannot plan in floor X
```
**可能原因**：
- 起点/终点在障碍物上
- 连接点选择无效

**解决**：重启程序，确保点击白色区域

### 程序崩溃
**解决**：
- 检查地图文件路径是否正确
- 查看控制台输出的图结构信息

## 项目结构

```
PRM-3D/
├── lib/PRMMAP/
│   ├── PRMmap.h           # 核心头文件
│   ├── PRMmapcd.cpp       # 地图构建
│   ├── PRMcommon.cpp      # 通用函数
│   ├── PRMthpp.cpp        # 路径规划
│   ├── PRMmultifloor.h    # 跨楼层头文件
│   ├── PRMmultifloor.cpp  # 跨楼层实现
│   └── debug.cpp          # 调试函数
├── test/
│   └── testMultiFloor.cpp # 测试程序
├── CMakeLists.txt
└── MULTIFLOOR_README.md   # 本文档
```
