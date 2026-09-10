#!/usr/bin/env python3
"""Convert geometry_msgs/TwistStamped commands to geometry_msgs/Twist."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class TwistStampedToTwist(Node):
    def __init__(self) -> None:
        super().__init__("twist_stamped_to_twist")

        self.declare_parameter("input_topic", "/cmu/cmd_vel_stamped")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("drop_zero_twist", False)
        self.declare_parameter("zero_epsilon", 1e-6)
        self.declare_parameter("enable_vehicle_cmd_filter", False)
        self.declare_parameter("min_turning_radius", 0.0)
        self.declare_parameter("min_turning_linear_speed", 0.0)
        self.declare_parameter("allow_in_place_rotation", False)
        self.declare_parameter("turning_radius_epsilon", 1e-6)
        self.declare_parameter("enable_odom_watchdog", False)
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("odom_timeout_sec", 0.5)
        self.declare_parameter("odom_jump_distance_threshold", 1.0)
        self.declare_parameter("odom_jump_yaw_threshold", 1.0)
        self.declare_parameter("odom_jump_recovery_sec", 5.0)
        self.declare_parameter("odom_stale_timeout_sec", 1.0)
        self.declare_parameter("odom_stale_position_epsilon", 0.001)
        self.declare_parameter("odom_stale_yaw_epsilon", 0.001)
        self.declare_parameter("publish_zero_on_odom_error", True)
        self.declare_parameter("zero_publish_rate", 5.0)
        self.declare_parameter("log_throttle_sec", 2.0)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        self.drop_zero_twist = bool(self.get_parameter("drop_zero_twist").value)
        self.zero_epsilon = float(self.get_parameter("zero_epsilon").value)
        self.enable_vehicle_cmd_filter = bool(
            self.get_parameter("enable_vehicle_cmd_filter").value
        )
        self.min_turning_radius = max(
            0.0, float(self.get_parameter("min_turning_radius").value)
        )
        self.min_turning_linear_speed = max(
            0.0, float(self.get_parameter("min_turning_linear_speed").value)
        )
        self.allow_in_place_rotation = bool(
            self.get_parameter("allow_in_place_rotation").value
        )
        self.turning_radius_epsilon = max(
            0.0, float(self.get_parameter("turning_radius_epsilon").value)
        )
        self.enable_odom_watchdog = bool(
            self.get_parameter("enable_odom_watchdog").value
        )
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.odom_jump_distance_threshold = float(
            self.get_parameter("odom_jump_distance_threshold").value
        )
        self.odom_jump_yaw_threshold = float(
            self.get_parameter("odom_jump_yaw_threshold").value
        )
        self.odom_jump_recovery_sec = float(
            self.get_parameter("odom_jump_recovery_sec").value
        )
        self.odom_stale_timeout_sec = float(
            self.get_parameter("odom_stale_timeout_sec").value
        )
        self.odom_stale_position_epsilon = float(
            self.get_parameter("odom_stale_position_epsilon").value
        )
        self.odom_stale_yaw_epsilon = float(
            self.get_parameter("odom_stale_yaw_epsilon").value
        )
        self.publish_zero_on_odom_error = bool(
            self.get_parameter("publish_zero_on_odom_error").value
        )
        self.zero_publish_rate = float(self.get_parameter("zero_publish_rate").value)
        self.log_throttle_sec = float(self.get_parameter("log_throttle_sec").value)
        self.last_valid_odom_time = None
        self.last_odom_issue = "no valid odometry received yet"
        self.last_odom_safe = True
        self.last_safe_odom_pose = None
        self.stale_reference_pose = None
        self.stale_reference_time = None
        self.jump_recovery_start_time = None
        self.jump_recovery_last_pose = None
        self.jump_recovery_stale_reference_pose = None
        self.jump_recovery_stale_reference_time = None
        self.last_log_time = 0.0
        self.last_log_message = ""
        self.last_vehicle_cmd_direction = 1.0

        self.pub = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(TwistStamped, input_topic, self.cb, 10)
        if self.enable_odom_watchdog:
            self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
            if self.publish_zero_on_odom_error and self.zero_publish_rate > 0.0:
                self.create_timer(
                    1.0 / self.zero_publish_rate, self.publish_zero_if_unsafe
                )

        self.get_logger().info(
            f"TwistStampedToTwist ready: {input_topic} -> {output_topic}, "
            f"drop_zero_twist={self.drop_zero_twist}, "
            f"vehicle_cmd_filter={self.enable_vehicle_cmd_filter}, "
            f"min_turning_radius={self.min_turning_radius:.3f}, "
            f"min_turning_linear_speed={self.min_turning_linear_speed:.3f}, "
            f"allow_in_place_rotation={self.allow_in_place_rotation}, "
            f"odom_watchdog={self.enable_odom_watchdog}, odom={odom_topic}"
        )

    def cb(self, msg: TwistStamped) -> None:
        if not self.is_twist_finite(msg.twist):
            self.warn_throttled("Blocking cmd_vel because input twist is non-finite.")
            self.publish_zero()
            return

        odom_ok, reason = self.odom_is_safe()
        if not odom_ok:
            self.warn_throttled(f"Blocking cmd_vel because {reason}.")
            self.publish_zero()
            return

        out = self.filtered_twist(msg.twist)

        if self.drop_zero_twist and self.is_zero(out):
            return

        self.pub.publish(out)

    def filtered_twist(self, twist: Twist) -> Twist:
        if not self.enable_vehicle_cmd_filter:
            out = Twist()
            out.linear = twist.linear
            out.angular = twist.angular
            return out

        out = Twist()
        out.linear.x = float(twist.linear.x)
        out.angular.z = float(twist.angular.z)
        if math.fabs(out.linear.x) > self.turning_radius_epsilon:
            self.last_vehicle_cmd_direction = math.copysign(1.0, out.linear.x)

        wz_abs = math.fabs(out.angular.z)
        if wz_abs <= self.turning_radius_epsilon:
            return out

        vx_abs = math.fabs(out.linear.x)
        if (
            self.min_turning_linear_speed > 0.0
            and vx_abs < self.min_turning_linear_speed
        ):
            direction = self.last_vehicle_cmd_direction
            if vx_abs > self.turning_radius_epsilon:
                direction = math.copysign(1.0, out.linear.x)
            out.linear.x = direction * self.min_turning_linear_speed
            vx_abs = self.min_turning_linear_speed

        if self.min_turning_radius <= 0.0:
            return out

        if vx_abs <= self.turning_radius_epsilon:
            if not self.allow_in_place_rotation:
                out.angular.z = 0.0
            return out

        max_wz = vx_abs / self.min_turning_radius
        if wz_abs > max_wz:
            out.angular.z = math.copysign(max_wz, out.angular.z)
        return out

    def odom_cb(self, msg: Odometry) -> None:
        if not self.is_odom_finite(msg):
            self.last_valid_odom_time = None
            self.last_odom_issue = "odometry pose contains non-finite values"
            self.last_safe_odom_pose = None
            self.stale_reference_pose = None
            self.stale_reference_time = None
            self.reset_jump_recovery()
            self.set_odom_safe(False, self.last_odom_issue)
            self.publish_zero()
            return

        now = time.monotonic()
        pose = self.odom_pose(msg)
        issue = self.check_odom_pose_health(pose, now)

        self.last_valid_odom_time = now
        self.last_odom_issue = issue or ""
        if issue is None:
            self.last_safe_odom_pose = pose

        odom_ok, reason = self.odom_is_safe()
        self.set_odom_safe(odom_ok, reason)
        if not odom_ok:
            self.publish_zero()

    def odom_is_safe(self):
        if not self.enable_odom_watchdog:
            return True, ""

        if self.last_valid_odom_time is None:
            return False, self.last_odom_issue

        age = time.monotonic() - self.last_valid_odom_time
        if age > self.odom_timeout_sec:
            return False, f"odometry timed out for {age:.2f}s"

        if self.last_odom_issue:
            return False, self.last_odom_issue

        return True, ""

    def publish_zero_if_unsafe(self) -> None:
        odom_ok, reason = self.odom_is_safe()
        self.set_odom_safe(odom_ok, reason)
        if not odom_ok:
            self.publish_zero()

    def set_odom_safe(self, safe: bool, reason: str) -> None:
        if safe == self.last_odom_safe:
            return

        self.last_odom_safe = safe
        if safe:
            self.get_logger().info(
                "Odometry watchdog recovered; cmd_vel passthrough enabled."
            )
        else:
            self.warn_throttled(f"Odometry watchdog active: {reason}.")

    def publish_zero(self) -> None:
        if not self.publish_zero_on_odom_error:
            return
        self.pub.publish(Twist())

    def is_zero(self, twist: Twist) -> bool:
        values = (
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        )
        return all(math.fabs(value) <= self.zero_epsilon for value in values)

    def is_twist_finite(self, twist: Twist) -> bool:
        values = (
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        )
        return all(math.isfinite(value) for value in values)

    def is_odom_finite(self, msg: Odometry) -> bool:
        pose = msg.pose.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        return all(math.isfinite(value) for value in values)

    def odom_pose(self, msg: Odometry):
        pose = msg.pose.pose
        return (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            self.quaternion_yaw(pose.orientation),
        )

    def check_odom_pose_health(self, pose, now: float):
        jump_issue = self.detect_odom_jump(pose)
        if jump_issue is not None:
            return self.check_jump_recovery(pose, now, jump_issue)

        self.reset_jump_recovery()

        stale_issue = self.detect_stale_odom(pose, now)
        if stale_issue is not None:
            return stale_issue

        return None

    def check_jump_recovery(self, pose, now: float, jump_issue: str):
        if self.odom_jump_recovery_sec <= 0.0:
            self.reset_jump_recovery()
            return jump_issue

        if self.jump_recovery_start_time is None:
            self.start_jump_recovery(pose, now)
            return self.jump_recovery_message(jump_issue, 0.0)

        candidate_jump_issue = self.detect_pose_jump(
            pose, self.jump_recovery_last_pose, "recovery odometry"
        )
        if candidate_jump_issue is not None:
            self.start_jump_recovery(pose, now)
            return f"{jump_issue}; recovery reset because {candidate_jump_issue}"

        self.jump_recovery_last_pose = pose
        candidate_stale_issue = self.detect_jump_recovery_stale(pose, now)
        if candidate_stale_issue is not None:
            self.jump_recovery_start_time = now
            return f"{jump_issue}; {candidate_stale_issue}"

        stable_duration = now - self.jump_recovery_start_time
        if stable_duration >= self.odom_jump_recovery_sec:
            self.stale_reference_pose = pose
            self.stale_reference_time = now
            self.reset_jump_recovery()
            return None

        return self.jump_recovery_message(jump_issue, stable_duration)

    def start_jump_recovery(self, pose, now: float) -> None:
        self.jump_recovery_start_time = now
        self.jump_recovery_last_pose = pose
        self.jump_recovery_stale_reference_pose = pose
        self.jump_recovery_stale_reference_time = now

    def reset_jump_recovery(self) -> None:
        self.jump_recovery_start_time = None
        self.jump_recovery_last_pose = None
        self.jump_recovery_stale_reference_pose = None
        self.jump_recovery_stale_reference_time = None

    def jump_recovery_message(self, jump_issue: str, stable_duration: float) -> str:
        return (
            f"{jump_issue}; recovery stable for {stable_duration:.2f}s/"
            f"{self.odom_jump_recovery_sec:.2f}s"
        )

    def detect_jump_recovery_stale(self, pose, now: float):
        if self.odom_stale_timeout_sec <= 0.0:
            self.jump_recovery_stale_reference_pose = pose
            self.jump_recovery_stale_reference_time = now
            return None

        if (
            self.jump_recovery_stale_reference_pose is None
            or self.jump_recovery_stale_reference_time is None
        ):
            self.jump_recovery_stale_reference_pose = pose
            self.jump_recovery_stale_reference_time = now
            return None

        distance = self.pose_distance(pose, self.jump_recovery_stale_reference_pose)
        yaw_delta = self.yaw_delta(pose[3], self.jump_recovery_stale_reference_pose[3])
        position_unchanged = distance <= self.odom_stale_position_epsilon
        yaw_unchanged = yaw_delta <= self.odom_stale_yaw_epsilon

        if not (position_unchanged and yaw_unchanged):
            self.jump_recovery_stale_reference_pose = pose
            self.jump_recovery_stale_reference_time = now
            return None

        stale_duration = now - self.jump_recovery_stale_reference_time
        if stale_duration >= self.odom_stale_timeout_sec:
            self.jump_recovery_stale_reference_pose = pose
            self.jump_recovery_stale_reference_time = now
            return (
                "recovery odometry pose has not changed for "
                f"{stale_duration:.2f}s"
            )

        return None

    def detect_odom_jump(self, pose):
        if self.last_safe_odom_pose is None:
            return None

        return self.detect_pose_jump(pose, self.last_safe_odom_pose, "odometry")

    def detect_pose_jump(self, pose, reference_pose, label: str):
        if reference_pose is None:
            return None

        distance = self.pose_distance(pose, reference_pose)
        yaw_delta = self.yaw_delta(pose[3], reference_pose[3])

        if (
            self.odom_jump_distance_threshold > 0.0
            and distance > self.odom_jump_distance_threshold
        ):
            return (
                f"{label} position jumped "
                f"{distance:.3f}m > {self.odom_jump_distance_threshold:.3f}m"
            )

        if (
            self.odom_jump_yaw_threshold > 0.0
            and yaw_delta > self.odom_jump_yaw_threshold
        ):
            return (
                f"{label} yaw jumped "
                f"{yaw_delta:.3f}rad > {self.odom_jump_yaw_threshold:.3f}rad"
            )

        return None

    def detect_stale_odom(self, pose, now: float):
        if self.odom_stale_timeout_sec <= 0.0:
            self.stale_reference_pose = pose
            self.stale_reference_time = now
            return None

        if self.stale_reference_pose is None or self.stale_reference_time is None:
            self.stale_reference_pose = pose
            self.stale_reference_time = now
            return None

        distance = self.pose_distance(pose, self.stale_reference_pose)
        yaw_delta = self.yaw_delta(pose[3], self.stale_reference_pose[3])
        position_unchanged = distance <= self.odom_stale_position_epsilon
        yaw_unchanged = yaw_delta <= self.odom_stale_yaw_epsilon

        if not (position_unchanged and yaw_unchanged):
            self.stale_reference_pose = pose
            self.stale_reference_time = now
            return None

        stale_duration = now - self.stale_reference_time
        if stale_duration >= self.odom_stale_timeout_sec:
            return (
                "odometry pose has not changed for "
                f"{stale_duration:.2f}s"
            )

        return None

    def quaternion_yaw(self, quat) -> float:
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def pose_distance(self, pose_a, pose_b) -> float:
        return math.sqrt(
            (pose_a[0] - pose_b[0]) ** 2
            + (pose_a[1] - pose_b[1]) ** 2
            + (pose_a[2] - pose_b[2]) ** 2
        )

    def yaw_delta(self, yaw_a: float, yaw_b: float) -> float:
        return math.fabs(
            math.atan2(math.sin(yaw_a - yaw_b), math.cos(yaw_a - yaw_b))
        )

    def warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if (
            message != self.last_log_message
            or now - self.last_log_time >= self.log_throttle_sec
        ):
            self.get_logger().warn(message)
            self.last_log_message = message
            self.last_log_time = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TwistStampedToTwist()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
