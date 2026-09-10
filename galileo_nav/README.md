# galileo_nav（精简版）

从 `galileo_nav_stack` 抽出的独立导航启动包，只保留 PRM 多楼层全局规划 + cmu/replan/wzh 局部规划及其桥接链。

- 全局规划：`global:=prm` → `prm_global_planner`（C++，PRM-3D 多楼层路图）
- 局部规划：`local:=cmu`（CMU 循迹）/ `local:=replan`（循迹 + 20Hz 碰撞监控重规划）/ `local:=wzh`（CMU 循迹 + 按楼层标签挂碰撞监控）
- 重规划服务接口：`prm_interfaces/ReplanPlan`

节点入口见 `setup.py` 的 `entry_points`。运行入口见仓库根 `prm_dow_ws/README.md`。
