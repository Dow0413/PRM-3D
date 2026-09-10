#!/usr/bin/env python3
"""Log odometry, waypoint, waypoint distances, and cmd_vel command values."""

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class TopicStateLogger(Node):
    def __init__(self) -> None:
        super().__init__("topic_state_logger")

        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("log_rate", 1.0)

        self.odom_topic = (
            str(self.get_parameter("odom_topic").value).strip() or "/Odometry"
        )
        self.cmd_vel_topic = (
            str(self.get_parameter("cmd_vel_topic").value).strip() or "/cmd_vel"
        )
        self.waypoint_topic = (
            str(self.get_parameter("waypoint_topic").value).strip() or "/way_point"
        )
        self.log_rate = max(float(self.get_parameter("log_rate").value), 0.1)

        self.latest_odom: Optional[Odometry] = None
        self.latest_cmd_vel: Optional[Twist] = None
        self.latest_waypoint: Optional[PointStamped] = None

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 20)
        self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, 20)
        self.create_subscription(
            PointStamped, self.waypoint_topic, self.waypoint_callback, 20
        )
        self.create_timer(1.0 / self.log_rate, self.log_state)

        self.get_logger().info(
            f"Topic state logger ready: odom={self.odom_topic}, "
            f"cmd_vel={self.cmd_vel_topic}, waypoint={self.waypoint_topic}, "
            f"log_rate={self.log_rate:.2f} Hz"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.latest_cmd_vel = msg

    def waypoint_callback(self, msg: PointStamped) -> None:
        self.latest_waypoint = msg

    def log_state(self) -> None:
        odom_text = "odom=waiting"
        cmd_text = "cmd_vel=waiting"
        waypoint_text = "way_point=waiting"
        distance_xoy_text = "distance_xoy=waiting"
        distance_3d_text = "distance_3d=waiting"

        if self.latest_odom is not None:
            position = self.latest_odom.pose.pose.position
            odom_text = (
                f"odom[{self.odom_topic}]: "
                f"x={position.x:.3f}, y={position.y:.3f}, z={position.z:.3f}"
            )

        if self.latest_cmd_vel is not None:
            linear = self.latest_cmd_vel.linear
            angular = self.latest_cmd_vel.angular
            cmd_text = (
                f"cmd_vel[{self.cmd_vel_topic}]: "
                f"v_x={linear.x:.3f}, v_y={linear.y:.3f}, w_z={angular.z:.3f}"
            )

        if self.latest_waypoint is not None:
            point = self.latest_waypoint.point
            waypoint_text = (
                f"way_point[{self.waypoint_topic}]: "
                f"x={point.x:.3f}, y={point.y:.3f}, z={point.z:.3f}"
            )

        if self.latest_odom is not None and self.latest_waypoint is not None:
            position = self.latest_odom.pose.pose.position
            point = self.latest_waypoint.point
            dx = point.x - position.x
            dy = point.y - position.y
            dz = point.z - position.z
            distance_xoy = math.hypot(dx, dy)
            distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            distance_xoy_text = (
                f"distance_xoy[{self.odom_topic}->{self.waypoint_topic}]: "
                f"{distance_xoy:.3f}"
            )
            distance_3d_text = (
                f"distance_3d[{self.odom_topic}->{self.waypoint_topic}]: "
                f"{distance_3d:.3f}"
            )

        self.get_logger().info(
            f"{odom_text}; {cmd_text}; {waypoint_text}; "
            f"{distance_xoy_text}; {distance_3d_text}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TopicStateLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
