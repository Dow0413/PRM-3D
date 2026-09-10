#!/usr/bin/env python3
"""Mux CMU cmd_vel with a short-lived final-alignment override."""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelMux(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_mux")

        self.declare_parameter("fallback_topic", "/cmd_vel/cmu")
        self.declare_parameter("override_topic", "/cmd_vel/final_align")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("override_timeout_sec", 0.35)

        self.fallback_topic = str(self.get_parameter("fallback_topic").value)
        self.override_topic = str(self.get_parameter("override_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.override_timeout_sec = max(
            0.0, float(self.get_parameter("override_timeout_sec").value)
        )
        self.last_override_time = 0.0

        self.pub = self.create_publisher(Twist, self.output_topic, 10)
        self.create_subscription(Twist, self.fallback_topic, self.fallback_cb, 10)
        self.create_subscription(Twist, self.override_topic, self.override_cb, 10)

        self.get_logger().info(
            "CmdVelMux ready: "
            f"fallback={self.fallback_topic}, override={self.override_topic}, "
            f"output={self.output_topic}, "
            f"override_timeout_sec={self.override_timeout_sec:.2f}"
        )

    def fallback_cb(self, msg: Twist) -> None:
        if self.override_active():
            return
        self.pub.publish(msg)

    def override_cb(self, msg: Twist) -> None:
        self.last_override_time = time.monotonic()
        self.pub.publish(msg)

    def override_active(self) -> bool:
        if self.last_override_time <= 0.0:
            return False
        return time.monotonic() - self.last_override_time <= self.override_timeout_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelMux()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
