#!/usr/bin/env python3
"""wzh 局部规划器监控节点：CMU 循迹 + 按楼层 PNG 标签（local_way）挂载碰撞监控。

wzh 架构的局部端（local:=wzh）：

- 循迹 100% 由 CMU 栈负责（local_cmu.launch.py：pathFollower/terrainAnalysis/
  localPlanner 等），本节点**不发任何 cmd_vel**——control_cb 被覆写为空操作。
- 本节点只做 local_replan 的那半边：障碍点云过滤 + 前视碰撞检测 +
  ReplanPlan 重规划 + 障碍持续入图，逻辑与 local_replan 完全一致（直接继承
  LocalReplan 复用）。
- 楼层门控改由 local_wzh.yaml maps.<env>.floors 的 ``local_way`` 标签决定
  （每个 PNG 一个）：
      local_way: replan -> 碰撞监控开（触发重规划 + 障碍入图，同 local_replan）
      local_way: cmu    -> 碰撞监控全关（纯 CMU 循迹：不触发、不入图）
  标签展开成 floor_local_ways 数组由 local_wzh.launch.py 传入；缺失/不识别的
  值按 replan 处理（与 enable_replan 默认 true 的惯例一致）。

与 local_replan 的两个语义差别（继承代码里 hold/停车保持 相关的日志措辞
在此不生效，特此说明）：

1. 本节点没有「停车保持」的执行手段——CMU 在驱动。真障碍挡路时 CMU 自己
   会停（局部路径无解 -> 不给前进速度）；重规划成功后全局端发布新
   /global_path，path_to_waypoint 桥自动喂新 waypoint，CMU 恢复。若以后需要
   强制停车，可向 CMU 栈的 /stop 话题发消息（pathFollower.cpp safetyStop），
   本版未实现。
2. 楼层切换（Z 带判定）复用基类 _floor_of_z，跨层任务中标签随机器人所在
   层实时切换，楼梯 PNG 打 cmu 标签即可整段关闭监控。
"""

import math

import rclpy
from rclpy.parameter import Parameter

from galileo_nav.local_replan import LocalReplan


class WzhReplan(LocalReplan):
    """local_replan 的监控半边：无循迹、无 cmd_vel，按 local_way 门控楼层。"""

    def __init__(self) -> None:
        super().__init__(node_name="wzh_replan")

        # 楼层标签（与 floor_z_mins 等并行数组）：replan=监控开，cmu=纯循迹。
        self.declare_parameter("floor_local_ways", Parameter.Type.STRING_ARRAY)
        raw = self.get_parameter("floor_local_ways").value or []
        ways = [str(v).strip().lower() for v in raw]

        if ways and len(ways) == len(self.floor_z_mins):
            # cmu -> False（该层监控全关：不触发、不入图），其余（replan/
            # 缺失/拼写异常）-> True，与 enable_replan 默认 true 的惯例一致。
            self.floor_enables = [w != "cmu" for w in ways]
            for i, (png, way) in enumerate(zip(self.floor_pngs, ways)):
                self.get_logger().info(
                    f"[wzh] F{i} ({png}): local_way={way or '(默认 replan)'} -> "
                    f"{'碰撞监控+入图' if self.floor_enables[i] else '纯 CMU 循迹'}"
                )
        elif not ways:
            self.get_logger().warn(
                "[wzh] 未收到 floor_local_ways 参数，全部楼层沿用 enable_replan "
                "开关（等同 local_replan 行为）"
            )
        else:
            self.get_logger().error(
                f"[wzh] floor_local_ways 长度 {len(ways)} 与楼层数 "
                f"{len(self.floor_z_mins)} 不一致，沿用 enable_replan 开关"
            )

    def control_cb(self) -> None:
        # wzh 不发 cmd_vel（循迹由 CMU 栈负责）。基类的 pub 对象仍在但不发布，
        # launch 侧已把 cmd_topic 指到死话题，双保险避免与 CMU 抢 /cmd_vel。
        # 基类的「到终点置 done」在 compute_cmd 里，wzh 不跑循迹永远不会触发，
        # 监控会在到达终点后一直空转（终点附近有障碍还会误触发）——在这里补
        # 终点锁存：XY < goal_tol 且 Z < z_tol 时置 done，_collision_check 随之休眠。
        if self.x is not None and not self.done and self.points:
            goal = self.points[-1]
            dxy = math.hypot(goal[0] - self.x, goal[1] - self.y)
            dz = abs(goal[2] - self.z)
            if dxy < self.goal_tol and dz < self.z_tol:
                self.done = True
                self.get_logger().info("[wzh] 到达终点, 碰撞监控休眠")
        return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WzhReplan()
    node.start_collision_thread()
    try:
        rclpy.spin(node)
    finally:
        node.stop_collision_thread()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
