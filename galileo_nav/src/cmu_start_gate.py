#!/usr/bin/env python3
"""Hold CMU pathFollower stopped until an active waypoint/path command exists."""

import math
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Int8


class CmuStartGate(Node):
    def __init__(self) -> None:
        super().__init__("cmu_start_gate")

        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("global_path_topic", "/global_path")
        self.declare_parameter("odom_topic", "/state_estimation")
        self.declare_parameter("stop_topic", "/stop")
        self.declare_parameter("stop_value", 2)
        self.declare_parameter("release_value", 0)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("min_path_points", 2)
        self.declare_parameter("release_publish_duration", 2.0)
        self.declare_parameter("goal_reached_tolerance", 0.3)
        self.declare_parameter("new_goal_epsilon", 0.05)
        self.declare_parameter("stop_on_goal", True)
        self.declare_parameter("ignore_reached_goal_repeats", True)
        self.declare_parameter("ignore_cancelled_path_refresh", True)

        self.stop_value = int(self.get_parameter("stop_value").value)
        self.release_value = int(self.get_parameter("release_value").value)
        self.min_path_points = int(self.get_parameter("min_path_points").value)
        self.release_publish_duration = float(
            self.get_parameter("release_publish_duration").value
        )
        self.goal_reached_tolerance = float(
            self.get_parameter("goal_reached_tolerance").value
        )
        self.new_goal_epsilon = float(self.get_parameter("new_goal_epsilon").value)
        self.stop_on_goal = bool(self.get_parameter("stop_on_goal").value)
        self.ignore_reached_goal_repeats = bool(
            self.get_parameter("ignore_reached_goal_repeats").value
        )
        self.ignore_cancelled_path_refresh = bool(
            self.get_parameter("ignore_cancelled_path_refresh").value
        )
        publish_rate = float(self.get_parameter("publish_rate").value)

        waypoint_topic = str(self.get_parameter("waypoint_topic").value)
        global_path_topic = str(self.get_parameter("global_path_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        stop_topic = str(self.get_parameter("stop_topic").value)

        self.released = False
        self.release_time = None
        self.robot_pos: Optional[Tuple[float, float, float]] = None
        self.active_goal: Optional[Tuple[float, float, float]] = None
        self.reached_goal: Optional[Tuple[float, float, float]] = None
        self.path_mode = False

        self.stop_pub = self.create_publisher(Int8, stop_topic, 10)
        self.create_subscription(PointStamped, waypoint_topic, self.waypoint_cb, 10)
        self.create_subscription(Path, global_path_topic, self.path_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.timer = self.create_timer(
            max(1.0 / max(publish_rate, 1e-3), 1e-3), self.timer_cb
        )

        self.get_logger().info(
            "CMU start gate active: holding /stop until waypoint/path command, "
            f"stop_on_goal={self.stop_on_goal}, "
            f"ignore_reached_goal_repeats={self.ignore_reached_goal_repeats}, "
            f"ignore_cancelled_path_refresh={self.ignore_cancelled_path_refresh}."
        )

    def waypoint_cb(self, msg: PointStamped) -> None:
        if self.path_mode:
            return
        if not self.stop_on_goal:
            self.release("waypoint", refresh=True)
            return
        self.accept_goal(
            (float(msg.point.x), float(msg.point.y), float(msg.point.z)), "waypoint"
        )

    def path_cb(self, msg: Path) -> None:
        if len(msg.poses) >= self.min_path_points:
            p = msg.poses[-1].pose.position
            self.path_mode = True
            if not self.stop_on_goal:
                self.release("global path", refresh=True)
                return
            self.accept_goal((float(p.x), float(p.y), float(p.z)), "global path")

    def odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.robot_pos = (float(p.x), float(p.y), float(p.z))

    def accept_goal(self, goal: Tuple[float, float, float], source: str) -> None:
        if (
            self.reached_goal is not None
            and self.dist_xy(goal, self.reached_goal) <= self.new_goal_epsilon
        ):
            still_at_reached_goal = (
                self.robot_pos is None
                or self.dist_xy(self.robot_pos, self.reached_goal)
                <= self.goal_reached_tolerance
            )
            if self.ignore_reached_goal_repeats and still_at_reached_goal:
                return
            self.get_logger().info(
                f"CMU start gate accepting repeated reached {source} goal."
            )
        if (
            self.active_goal is not None
            and self.dist_xy(goal, self.active_goal) <= self.new_goal_epsilon
        ):
            if self.released:
                if (
                    source == "global path"
                    and not self.ignore_cancelled_path_refresh
                ):
                    self.release(source, refresh=True)
                    return
                return

        self.active_goal = goal
        self.reached_goal = None
        self.release(source)

    def release(self, source: str, refresh: bool = False) -> None:
        if self.released and not refresh:
            return
        was_released = self.released
        self.released = True
        self.release_time = time.monotonic()
        if was_released and refresh:
            self.get_logger().info(f"CMU start gate release refreshed by {source}.")
        else:
            self.get_logger().info(f"CMU start gate released by {source}.")

    def timer_cb(self) -> None:
        if not self.released:
            self.stop_pub.publish(Int8(data=self.stop_value))
            return

        if self.active_goal is not None and self.robot_pos is not None:
            if (
                self.dist_xy(self.robot_pos, self.active_goal)
                <= self.goal_reached_tolerance
            ):
                self.reached_goal = self.active_goal
                self.active_goal = None
                self.path_mode = False
                self.released = False
                self.release_time = None
                self.stop_pub.publish(Int8(data=self.stop_value))
                self.get_logger().info("CMU active goal reached; navigation stopped.")
                return

        if self.release_time is None:
            return

        if time.monotonic() - self.release_time <= self.release_publish_duration:
            self.stop_pub.publish(Int8(data=self.release_value))

    @staticmethod
    def dist_xy(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmuStartGate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
