#!/usr/bin/env python3
"""Pure-pursuit follower: /global_path + odometry -> nav_vel Twist.

Bridges the PRM global path to the robot's velocity interface (by default
/cmd_vel).

Progress tracking and the goal check are 3D: the robot is projected onto the
path polyline with Z weighted (x2) so overlapping floors never hijack the
progress estimate, and the goal only counts when the robot is at the goal's
height as well as its XY — otherwise a multi-floor tour that passes above/below
the goal XY would stop early. Steering itself is 2D; the platform handles
terrain/stairs.
"""

import bisect
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node


class PathFollower(Node):
    def __init__(self, node_name: str = "path_follower") -> None:
        super().__init__(node_name)

        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("lookahead_distance", 0.5)
        self.declare_parameter("max_linear_velocity", 0.4)
        self.declare_parameter("max_angular_velocity", 0.8)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("floor_z_tolerance", 0.8)
        self.declare_parameter("slow_radius", 1.0)
        self.declare_parameter("rotate_cutoff_rad", 0.7)
        self.declare_parameter("backtrack_window", 2.0)
        self.declare_parameter("window_ahead", 8.0)
        self.declare_parameter("control_rate", 20.0)

        self.path_topic = str(self.get_parameter("path_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.lookahead = max(0.1, float(self.get_parameter("lookahead_distance").value))
        self.max_v = max(0.05, float(self.get_parameter("max_linear_velocity").value))
        self.max_w = max(0.1, float(self.get_parameter("max_angular_velocity").value))
        self.goal_tol = max(0.05, float(self.get_parameter("goal_tolerance").value))
        self.z_tol = max(0.2, float(self.get_parameter("floor_z_tolerance").value))
        self.slow_radius = max(0.1, float(self.get_parameter("slow_radius").value))
        self.rotate_cutoff = float(self.get_parameter("rotate_cutoff_rad").value)
        self.backtrack = max(0.0, float(self.get_parameter("backtrack_window").value))
        self.window_ahead = max(1.0, float(self.get_parameter("window_ahead").value))
        control_rate = max(1.0, float(self.get_parameter("control_rate").value))

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_subscription(Path, self.path_topic, self.path_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_timer(1.0 / control_rate, self.control_cb)

        self.points = []            # [(x, y, z)] in map frame
        self.cum = []               # cumulative 3D arc length at each point
        self.progress = 0           # index of the current nearest path point
        self.done = True
        self.hold = False           # 全局规划失败(空路径)→停车保持，等新路径恢复
        self.x = self.y = self.z = self.yaw = None

        self.get_logger().info(
            "PathFollower ready: "
            f"path={self.path_topic}, odom={self.odom_topic}, cmd={self.cmd_topic}, "
            f"lookahead={self.lookahead:.2f}, v_max={self.max_v:.2f}, "
            f"w_max={self.max_w:.2f}, goal_tol=({self.goal_tol:.2f}xy, "
            f"{self.z_tol:.2f}z)"
        )

    def path_cb(self, msg: Path) -> None:
        pts = [(p.pose.position.x, p.pose.position.y, p.pose.position.z)
               for p in msg.poses]
        if len(pts) < 2:
            # 空/退化路径 = 全局规划失败（如动态障碍封死所有通道，含降级重试后
            # 仍无路）。停车保持而不是继续跟旧路径撞上去；同时保留旧路径，让
            # local_replan 的触发器继续调用重规划服务，障碍过期(decay)或
            # 挪走后规划器会重新给出有效路径，收到即自动恢复行驶。
            if self.points and not self.done:
                self.hold = True
                self.get_logger().warn(
                    "[规划情况] 收到空路径 (全局规划失败) -> 停车保持, 等待可行路线")
            return
        self.hold = False
        self.points = pts
        self.cum = [0.0]
        for i in range(1, len(pts)):
            self.cum.append(self.cum[-1] + dist3(pts[i - 1], pts[i]))
        self.progress = 0
        self.done = False
        goal = pts[-1]
        self.get_logger().info(
            f"[规划情况] 新路径: {len(pts)} 点, 长 {self.cum[-1]:.1f} m, "
            f"终点 ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})"
        )

    def odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        self.x, self.y, self.z = pos.x, pos.y, pos.z
        self.yaw = self.extract_yaw(msg.pose.pose.orientation)

    @staticmethod
    def extract_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def control_cb(self) -> None:
        cmd = Twist()
        if self.hold:
            # 停车保持：持续发零速指令（覆盖可能残留的上一次非零指令）
            self.pub.publish(cmd)
            return
        if self.x is not None and not self.done and self.points:
            cmd = self.compute_cmd()
        self.pub.publish(cmd)

    def compute_cmd(self) -> Twist:
        s = self.update_progress()

        goal = self.points[-1]
        dxy_goal = math.hypot(goal[0] - self.x, goal[1] - self.y)
        dz_goal = abs(goal[2] - self.z)
        if dxy_goal < self.goal_tol and dz_goal < self.z_tol:
            self.done = True
            self.get_logger().info(
                f"[到达终点] ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}), 停车"
            )
            return Twist()

        target = self.lookahead_point(s + self.lookahead)
        dx, dy = target[0] - self.x, target[1] - self.y
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        tx = cos_y * dx + sin_y * dy            # target in body frame
        ty = -sin_y * dx + cos_y * dy
        alpha = math.atan2(ty, tx)

        # Slow near the goal and when off-heading (cos(alpha) -> 0), instead of
        # a hard v=0/rotate split that lurches at every corner. The ramp uses
        # the remaining arc length, NOT the XY distance to the goal: on
        # stacked floors the goal XY can sit right above/below the robot
        # (remaining route ~20 m) and an XY ramp would freeze it at v=0.
        remaining = self.cum[-1] - s
        speed_scale = min(1.0, remaining / self.slow_radius) * max(0.0, math.cos(alpha))
        if abs(alpha) > self.rotate_cutoff:
            v = 0.0                             # too far off heading: rotate in place
        else:
            v = self.max_v * speed_scale
        w = 2.0 * v * math.sin(alpha) / self.lookahead if v > 0.0 else 1.5 * alpha
        w = max(-self.max_w, min(self.max_w, w))

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        return cmd

    def update_progress(self) -> float:
        """Project the robot onto the path polyline within
        [progress - backtrack, progress + ahead] along the arc and return the
        projected arc position. A projection (not nearest vertex) is required:
        with sparse path segments the nearest vertex stalls at a segment end
        and the carrot never advances. Z is weighted x2 so segments on another
        floor that happen to share this XY never win."""
        base = self.cum[self.progress]
        lo = bisect.bisect_left(self.cum, base - self.backtrack)
        hi = min(bisect.bisect_right(self.cum, base + self.window_ahead),
                 len(self.points) - 1)
        best_s = self.cum[self.progress]
        best_i, best_d = self.progress, float("inf")
        for i in range(lo, hi):
            a, b = self.points[i], self.points[i + 1]
            abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            l2 = abx * abx + aby * aby + abz * abz
            if l2 < 1e-9:
                t = 0.0
            else:
                t = ((self.x - a[0]) * abx + (self.y - a[1]) * aby
                     + (self.z - a[2]) * abz) / l2
                t = max(0.0, min(1.0, t))
            qx, qy, qz = a[0] + t * abx, a[1] + t * aby, a[2] + t * abz
            d = math.hypot(qx - self.x, qy - self.y, 2.0 * (qz - self.z))
            if d < best_d:
                best_d = d
                best_s = self.cum[i] + t * math.sqrt(l2)
                best_i = i + 1 if t > 0.5 else i    # vertex anchoring the next window
        self.progress = best_i
        return best_s

    def lookahead_point(self, target_arc):
        i = bisect.bisect_left(self.cum, target_arc)
        if i >= len(self.points):
            return self.points[-1]
        if i == 0:
            return self.points[0]
        # interpolate between i-1 and i along the arc
        seg = self.cum[i] - self.cum[i - 1]
        t = (target_arc - self.cum[i - 1]) / seg if seg > 1e-6 else 0.0
        a, b = self.points[i - 1], self.points[i]
        return tuple(a[k] + t * (b[k] - a[k]) for k in range(3))


def dist3(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
