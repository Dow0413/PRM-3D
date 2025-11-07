## 📄 中文 README

# CDT-TetheredPathPlanning

这是一个用于2D系留机器人快速最优路径规划的库。本项目提供了多种高效的路径规划算法，适用于单目标、多目标访问等不同场景，并支持非同伦最优配置搜索。

## 🧩 功能模块

- **CDT-TCS**（Configuration Deformation Tree - Topological Configuration Search）  
  快速的非同伦最优配置搜索  
  实现于：`BImap::THPPtaskInit`

- **CDT-TPP**（Configuration Deformation Tree - Tethered Path Planner）  
  系留机器人最优路径规划  
  实现于：`BImap::THPPoptimalPlanner`

- **CDT-TMV**（Configuration Deformation Tree - Tethered Multi-Visit Planner）  
  系留机器人最优多目标访问路径规划  
  实现于：`BImap::TMVoptimalPlanner`

- **CDT-UTPP**（Configuration Deformation Tree - Untethered Path Planner）  
  非系留机器人最优路径规划  
  实现于：`BImap::UTHPPoptimalPlanner`

## ⚙️ 运行要求

### Python 部分
在运行测试程序前，请使用安装了 **OpenCV 3.4.9** 的 **Python3** 执行以下命令启动多边形拟合服务：
```bash
python3 approx_work.py
```

### C++ 编译依赖
- OpenCV 4.0 或更高版本

## 📁 示例工程说明

`./test/` 目录下包含7个示例程序：

- **无交互界面性能测试程序**：
  - `expXXX.cpp`

- **有交互界面测试程序**：
  - `testXXX.cpp`

## 🚀 使用方法

1. 启动 Python 多边形拟合服务：
   ```bash
   python3 approx_work.py
   ```

2. 编译 C++ 代码并运行相应的测试程序。

## 💡 注意事项

- 确保所有依赖项正确安装。
- 若需调试或扩展功能，请参考源码中的类与函数定义。
- 更多技术细节请参阅相关论文或文档（如有）。

---

## 🌐 English README

# CDT-TetheredPathPlanning

This is a library for fast and optimal path planning of 2D tethered robots. It includes several efficient algorithms suitable for various scenarios such as single-target, multi-target visiting, and topological configuration searching.

## 🧩 Functional Modules

- **CDT-TCS** (Configuration Deformation Tree - Topological Configuration Search)  
  Fast non-homotopic optimal configuration search  
  Implemented in: `BImap::THPPtaskInit`

- **CDT-TPP** (Configuration Deformation Tree - Tethered Path Planner)  
  Optimal path planning for tethered robots  
  Implemented in: `BImap::THPPoptimalPlanner`

- **CDT-TMV** (Configuration Deformation Tree - Tethered Multi-Visit Planner)  
  Optimal multi-target visiting planning for tethered robots  
  Implemented in: `BImap::TMVoptimalPlanner`

- **CDT-UTPP** (Configuration Deformation Tree - Untethered Path Planner)  
  Optimal path planning for untethered robots  
  Implemented in: `BImap::UTHPPoptimalPlanner`

## ⚙️ Requirements

### Python Part
Before running the test programs, please run the following command using **Python3** with **OpenCV 3.4.9** installed to start the polygon approximation service:
```bash
python3 approx_work.py
```

### C++ Build Dependencies
- OpenCV 4.0 or higher

## 📁 Example Projects

The `./test/` directory contains 7 example projects:

- **Performance test programs without UI**:
  - `expXXX.cpp`

- **Test programs with interactive UI**:
  - `testXXX.cpp`

## 🚀 Usage Instructions

1. Start the Python polygon approximation service:
   ```bash
   python3 approx_work.py
   ```

2. Compile the C++ code and run the corresponding test program.

## 💡 Notes

- Make sure all dependencies are correctly installed.
- For debugging or extending functionalities, refer to class and function definitions in the source code.
- For more technical details, please refer to related papers or documentation (if available).

---