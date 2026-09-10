#!/usr/bin/env python3
"""Post-process planar cmd_vel commands before they reach the robot SDK."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class CmdVelPostprocessor(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_postprocessor")

        self.declare_parameter("enabled", True)
        self.declare_parameter("input_topic", "/cmd_vel_nav")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("control_rate", 30.0)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("odom_timeout_sec", 0.5)
        self.declare_parameter("publish_zero_on_cmd_timeout", True)
        self.declare_parameter("preserve_unprocessed_axes", True)
        self.declare_parameter("zero_epsilon", 1e-5)
        self.declare_parameter("log_throttle_sec", 2.0)

        self.declare_parameter("enable_reachability_limit", True)
        self.declare_parameter("reachable_linear_accel", 0.6)
        self.declare_parameter("reachable_angular_accel", 1.2)

        self.declare_parameter("enable_pi_compensation", True)
        self.declare_parameter("require_fresh_odom_for_pi", True)
        self.declare_parameter("use_last_output_when_odom_stale", True)
        self.declare_parameter("feedforward_linear_gain", 1.0)
        self.declare_parameter("feedforward_angular_gain", 1.0)
        self.declare_parameter("linear_kp", 0.0)
        self.declare_parameter("linear_ki", 0.0)
        self.declare_parameter("linear_kd", 0.0)
        self.declare_parameter("angular_kp", 0.0)
        self.declare_parameter("angular_ki", 0.0)
        self.declare_parameter("angular_kd", 0.0)
        self.declare_parameter("linear_integral_limit", 0.5)
        self.declare_parameter("angular_integral_limit", 0.8)
        self.declare_parameter("linear_error_deadband", 0.01)
        self.declare_parameter("angular_error_deadband", 0.02)
        self.declare_parameter("reset_integral_on_target_sign_change", True)

        self.declare_parameter("enable_accel_limit", True)
        self.declare_parameter("max_linear_accel", 0.5)
        self.declare_parameter("max_linear_decel", 0.8)
        self.declare_parameter("max_angular_accel", 1.2)
        self.declare_parameter("max_angular_decel", 1.8)

        self.declare_parameter("enable_max_velocity_limit", True)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_reverse_speed", 0.5)
        self.declare_parameter("max_angular_speed", 0.872665)

        self.declare_parameter("enable_min_turning_radius", False)
        self.declare_parameter("min_turning_radius", 0.0)
        self.declare_parameter("min_turning_linear_speed", 0.0)
        self.declare_parameter("allow_in_place_rotation", False)
        self.declare_parameter("turning_radius_epsilon", 1e-6)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        if self.input_topic == self.output_topic:
            raise RuntimeError(
                "cmd_vel_postprocessor input_topic and output_topic "
                "must differ "
                "to avoid feeding its own output back into the filter."
            )

        self.control_rate = max(
            1e-3, float(self.get_parameter("control_rate").value)
        )
        self.command_timeout_sec = float(
            self.get_parameter("command_timeout_sec").value
        )
        self.odom_timeout_sec = float(
            self.get_parameter("odom_timeout_sec").value
        )
        self.publish_zero_on_cmd_timeout = bool(
            self.get_parameter("publish_zero_on_cmd_timeout").value
        )
        self.preserve_unprocessed_axes = bool(
            self.get_parameter("preserve_unprocessed_axes").value
        )
        self.zero_epsilon = max(
            0.0, float(self.get_parameter("zero_epsilon").value)
        )
        self.log_throttle_sec = max(
            0.0, float(self.get_parameter("log_throttle_sec").value)
        )

        self.enable_reachability_limit = bool(
            self.get_parameter("enable_reachability_limit").value
        )
        self.reachable_linear_accel = max(
            0.0, float(self.get_parameter("reachable_linear_accel").value)
        )
        self.reachable_angular_accel = max(
            0.0, float(self.get_parameter("reachable_angular_accel").value)
        )

        self.enable_pi_compensation = bool(
            self.get_parameter("enable_pi_compensation").value
        )
        self.require_fresh_odom_for_pi = bool(
            self.get_parameter("require_fresh_odom_for_pi").value
        )
        self.use_last_output_when_odom_stale = bool(
            self.get_parameter("use_last_output_when_odom_stale").value
        )
        self.feedforward_linear_gain = float(
            self.get_parameter("feedforward_linear_gain").value
        )
        self.feedforward_angular_gain = float(
            self.get_parameter("feedforward_angular_gain").value
        )
        self.linear_kp = float(self.get_parameter("linear_kp").value)
        self.linear_ki = float(self.get_parameter("linear_ki").value)
        self.linear_kd = float(self.get_parameter("linear_kd").value)
        self.angular_kp = float(self.get_parameter("angular_kp").value)
        self.angular_ki = float(self.get_parameter("angular_ki").value)
        self.angular_kd = float(self.get_parameter("angular_kd").value)
        self.linear_integral_limit = max(
            0.0, float(self.get_parameter("linear_integral_limit").value)
        )
        self.angular_integral_limit = max(
            0.0, float(self.get_parameter("angular_integral_limit").value)
        )
        self.linear_error_deadband = max(
            0.0, float(self.get_parameter("linear_error_deadband").value)
        )
        self.angular_error_deadband = max(
            0.0, float(self.get_parameter("angular_error_deadband").value)
        )
        self.reset_integral_on_target_sign_change = bool(
            self.get_parameter("reset_integral_on_target_sign_change").value
        )

        self.enable_accel_limit = bool(
            self.get_parameter("enable_accel_limit").value
        )
        self.max_linear_accel = max(
            0.0, float(self.get_parameter("max_linear_accel").value)
        )
        self.max_linear_decel = max(
            0.0, float(self.get_parameter("max_linear_decel").value)
        )
        self.max_angular_accel = max(
            0.0, float(self.get_parameter("max_angular_accel").value)
        )
        self.max_angular_decel = max(
            0.0, float(self.get_parameter("max_angular_decel").value)
        )

        self.enable_max_velocity_limit = bool(
            self.get_parameter("enable_max_velocity_limit").value
        )
        self.max_linear_speed = max(
            0.0, float(self.get_parameter("max_linear_speed").value)
        )
        self.max_reverse_speed = max(
            0.0, float(self.get_parameter("max_reverse_speed").value)
        )
        self.max_angular_speed = max(
            0.0, float(self.get_parameter("max_angular_speed").value)
        )

        self.enable_min_turning_radius = bool(
            self.get_parameter("enable_min_turning_radius").value
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

        self.last_desired_cmd = None
        self.last_desired_time = None
        self.last_measured_velocity = None
        self.last_odom_time = None
        self.last_output = Twist()
        self.last_control_time = None
        self.last_turning_direction = 1.0
        self.integral_linear = 0.0
        self.integral_angular = 0.0
        self.last_linear_error = None
        self.last_angular_error = None
        self.last_linear_target_sign = 0
        self.last_angular_target_sign = 0
        self.last_log_time = 0.0
        self.last_log_message = ""

        self.pub = self.create_publisher(Twist, self.output_topic, 10)
        self.create_subscription(Twist, self.input_topic, self.cmd_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_timer(1.0 / self.control_rate, self.timer_cb)

        self.get_logger().info(
            "CmdVelPostprocessor ready: "
            f"{self.input_topic} -> {self.output_topic}, "
            f"odom={self.odom_topic}, "
            f"enabled={self.enabled}, rate={self.control_rate:.1f}Hz, "
            f"pid=({self.linear_kp:.3f},{self.linear_ki:.3f},"
            f"{self.linear_kd:.3f})/({self.angular_kp:.3f},"
            f"{self.angular_ki:.3f},{self.angular_kd:.3f})"
        )

    def cmd_cb(self, msg: Twist) -> None:
        if not self.is_twist_finite(msg):
            self.warn_throttled("Dropping non-finite input cmd_vel.")
            return
        self.last_desired_cmd = self.copy_twist(msg)
        self.last_desired_time = time.monotonic()

    def odom_cb(self, msg: Odometry) -> None:
        twist = msg.twist.twist
        values = (twist.linear.x, twist.angular.z)
        if not all(math.isfinite(value) for value in values):
            self.warn_throttled(
                "Ignoring odometry with non-finite twist values."
            )
            self.last_measured_velocity = None
            self.last_odom_time = None
            return

        self.last_measured_velocity = (
            float(twist.linear.x),
            float(twist.angular.z),
        )
        self.last_odom_time = time.monotonic()

    def timer_cb(self) -> None:
        now = time.monotonic()
        dt = self.control_dt(now)

        if self.last_desired_cmd is None or self.last_desired_time is None:
            return

        if self.command_is_stale(now):
            self.reset_integrators()
            if self.publish_zero_on_cmd_timeout:
                self.publish_output(Twist())
            return

        if not self.enabled:
            self.reset_integrators()
            self.publish_output(self.last_desired_cmd)
            return

        has_fresh_odom = self.odom_is_fresh(now)
        measured_linear, measured_angular = self.measured_velocity(
            has_fresh_odom
        )
        if (
            self.enable_pi_compensation
            and self.require_fresh_odom_for_pi
            and not has_fresh_odom
        ):
            self.warn_throttled(
                "Odometry is stale; publishing feedforward command "
                "without PID correction."
            )

        out = self.process_cmd(
            self.last_desired_cmd,
            measured_linear,
            measured_angular,
            has_fresh_odom,
            dt,
        )
        self.publish_output(out)

    def control_dt(self, now: float) -> float:
        nominal_dt = 1.0 / self.control_rate
        if self.last_control_time is None:
            self.last_control_time = now
            return nominal_dt

        dt = now - self.last_control_time
        self.last_control_time = now
        if dt <= 0.0:
            return nominal_dt
        return min(dt, max(0.25, 5.0 * nominal_dt))

    def command_is_stale(self, now: float) -> bool:
        if self.command_timeout_sec <= 0.0:
            return False
        return now - self.last_desired_time > self.command_timeout_sec

    def odom_is_fresh(self, now: float) -> bool:
        if self.last_odom_time is None:
            return False
        if self.odom_timeout_sec <= 0.0:
            return True
        return now - self.last_odom_time <= self.odom_timeout_sec

    def measured_velocity(self, has_fresh_odom: bool):
        if has_fresh_odom and self.last_measured_velocity is not None:
            return self.last_measured_velocity
        if self.use_last_output_when_odom_stale:
            return self.last_output.linear.x, self.last_output.angular.z
        return 0.0, 0.0

    def process_cmd(
        self,
        desired: Twist,
        measured_linear: float,
        measured_angular: float,
        has_fresh_odom: bool,
        dt: float,
    ) -> Twist:
        target_linear = float(desired.linear.x)
        target_angular = float(desired.angular.z)

        if self.enable_reachability_limit:
            target_linear = self.limit_reachable(
                target_linear, measured_linear, self.reachable_linear_accel, dt
            )
            target_angular = self.limit_reachable(
                target_angular,
                measured_angular,
                self.reachable_angular_accel,
                dt,
            )

        can_apply_pid = (
            self.enable_pi_compensation
            and (has_fresh_odom or not self.require_fresh_odom_for_pi)
        )
        if can_apply_pid:
            out_linear = self.pid_axis(
                "linear",
                target_linear,
                measured_linear,
                self.feedforward_linear_gain,
                self.linear_kp,
                self.linear_ki,
                self.linear_kd,
                self.linear_integral_limit,
                self.linear_error_deadband,
                dt,
            )
            out_angular = self.pid_axis(
                "angular",
                target_angular,
                measured_angular,
                self.feedforward_angular_gain,
                self.angular_kp,
                self.angular_ki,
                self.angular_kd,
                self.angular_integral_limit,
                self.angular_error_deadband,
                dt,
            )
        else:
            out_linear = target_linear
            out_angular = target_angular
            self.reset_integrators()

        out = self.base_output_twist(desired)
        out.linear.x = out_linear
        out.angular.z = out_angular

        if self.enable_accel_limit:
            out.linear.x = self.limit_acceleration(
                out.linear.x,
                self.last_output.linear.x,
                self.max_linear_accel,
                self.max_linear_decel,
                dt,
            )
            out.angular.z = self.limit_acceleration(
                out.angular.z,
                self.last_output.angular.z,
                self.max_angular_accel,
                self.max_angular_decel,
                dt,
            )

        if self.enable_max_velocity_limit:
            out.linear.x = self.limit_linear_speed(out.linear.x)
            out.angular.z = self.limit_abs(
                out.angular.z, self.max_angular_speed
            )

        if self.enable_min_turning_radius:
            out = self.apply_min_turning_radius(out)

        if self.is_zero(out):
            self.reset_integrators()

        return out

    def limit_reachable(
        self, target: float, measured: float, accel_limit: float, dt: float
    ) -> float:
        if accel_limit <= 0.0:
            return target
        max_delta = accel_limit * dt
        return self.clamp(target, measured - max_delta, measured + max_delta)

    def pid_axis(
        self,
        axis: str,
        target: float,
        measured: float,
        feedforward_gain: float,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float,
        error_deadband: float,
        dt: float,
    ) -> float:
        error = target - measured
        sign = self.sign(target)
        last_sign = (
            self.last_linear_target_sign
            if axis == "linear"
            else self.last_angular_target_sign
        )
        if (
            self.reset_integral_on_target_sign_change
            and sign != 0
            and last_sign != 0
            and sign != last_sign
        ):
            self.set_integral(axis, 0.0)
            self.set_last_error(axis, None)

        if sign == 0 and math.fabs(measured) <= self.zero_epsilon:
            self.set_integral(axis, 0.0)
            self.set_last_error(axis, None)
        elif ki != 0.0:
            integral_error = (
                0.0 if math.fabs(error) <= error_deadband else error
            )
            integral = self.get_integral(axis) + integral_error * dt
            self.set_integral(axis, self.limit_abs(integral, integral_limit))

        derivative = self.error_derivative(axis, error, error_deadband, dt)

        if axis == "linear":
            self.last_linear_target_sign = sign
        else:
            self.last_angular_target_sign = sign

        return (
            feedforward_gain * target
            + kp * error
            + ki * self.get_integral(axis)
            + kd * derivative
        )

    def limit_acceleration(
        self,
        target: float,
        previous: float,
        accel_limit: float,
        decel_limit: float,
        dt: float,
    ) -> float:
        delta = target - previous
        if math.fabs(delta) <= self.zero_epsilon:
            return target

        if math.fabs(previous) <= self.zero_epsilon:
            reducing_speed = False
        elif previous * target < 0.0:
            reducing_speed = True
        else:
            reducing_speed = math.fabs(target) < math.fabs(previous)
        limit = decel_limit if reducing_speed else accel_limit
        if limit <= 0.0:
            return target

        max_delta = limit * dt
        return previous + self.clamp(delta, -max_delta, max_delta)

    def limit_linear_speed(self, value: float) -> float:
        reverse_limit = self.max_reverse_speed
        if reverse_limit <= 0.0:
            reverse_limit = self.max_linear_speed
        if self.max_linear_speed > 0.0:
            value = min(value, self.max_linear_speed)
        if reverse_limit > 0.0:
            value = max(value, -reverse_limit)
        return value

    def apply_min_turning_radius(self, twist: Twist) -> Twist:
        out = self.copy_twist(twist)
        if math.fabs(out.linear.x) > self.turning_radius_epsilon:
            self.last_turning_direction = math.copysign(1.0, out.linear.x)

        angular_abs = math.fabs(out.angular.z)
        if angular_abs <= self.turning_radius_epsilon:
            return out

        linear_abs = math.fabs(out.linear.x)
        if (
            self.min_turning_linear_speed > 0.0
            and linear_abs < self.min_turning_linear_speed
        ):
            direction = self.last_turning_direction
            if linear_abs > self.turning_radius_epsilon:
                direction = math.copysign(1.0, out.linear.x)
            out.linear.x = direction * self.min_turning_linear_speed
            linear_abs = self.min_turning_linear_speed

        if self.min_turning_radius <= 0.0:
            return out

        if linear_abs <= self.turning_radius_epsilon:
            if not self.allow_in_place_rotation:
                out.angular.z = 0.0
            return out

        max_angular = linear_abs / self.min_turning_radius
        if angular_abs > max_angular:
            out.angular.z = math.copysign(max_angular, out.angular.z)
        return out

    def publish_output(self, msg: Twist) -> None:
        self.pub.publish(msg)
        self.last_output = self.copy_twist(msg)

    def reset_integrators(self) -> None:
        self.integral_linear = 0.0
        self.integral_angular = 0.0
        self.last_linear_error = None
        self.last_angular_error = None
        self.last_linear_target_sign = 0
        self.last_angular_target_sign = 0

    def get_integral(self, axis: str) -> float:
        if axis == "linear":
            return self.integral_linear
        return self.integral_angular

    def set_integral(self, axis: str, value: float) -> None:
        if axis == "linear":
            self.integral_linear = value
        else:
            self.integral_angular = value

    def error_derivative(
        self, axis: str, error: float, error_deadband: float, dt: float
    ) -> float:
        derivative_error = 0.0 if math.fabs(error) <= error_deadband else error
        last_error = self.get_last_error(axis)
        self.set_last_error(axis, derivative_error)
        if last_error is None or dt <= 0.0:
            return 0.0
        return (derivative_error - last_error) / dt

    def get_last_error(self, axis: str):
        if axis == "linear":
            return self.last_linear_error
        return self.last_angular_error

    def set_last_error(self, axis: str, value) -> None:
        if axis == "linear":
            self.last_linear_error = value
        else:
            self.last_angular_error = value

    def base_output_twist(self, desired: Twist) -> Twist:
        if self.preserve_unprocessed_axes:
            return self.copy_twist(desired)
        return Twist()

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

    def copy_twist(self, twist: Twist) -> Twist:
        out = Twist()
        out.linear.x = float(twist.linear.x)
        out.linear.y = float(twist.linear.y)
        out.linear.z = float(twist.linear.z)
        out.angular.x = float(twist.angular.x)
        out.angular.y = float(twist.angular.y)
        out.angular.z = float(twist.angular.z)
        return out

    def sign(self, value: float) -> int:
        if value > self.zero_epsilon:
            return 1
        if value < -self.zero_epsilon:
            return -1
        return 0

    def clamp(self, value: float, low: float, high: float) -> float:
        return min(max(value, low), high)

    def limit_abs(self, value: float, limit: float) -> float:
        if limit <= 0.0:
            return value
        return self.clamp(value, -limit, limit)

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
    node = CmdVelPostprocessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
