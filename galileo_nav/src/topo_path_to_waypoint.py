#!/usr/bin/env python3
"""Bridge global Path to CMU local_planner /way_point + /speed interface."""

from dataclasses import dataclass
import math
import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Int8
from topo_nav_interfaces.msg import TopoPathMeta


@dataclass
class PathProjection:
    """Robot projection result on one path segment."""

    segment_index: int
    segment_t: float
    path_s: float
    point: Tuple[float, float, float]
    distance: float


class TopoPathToWaypoint(Node):
    """Convert global Path + odom into CMU waypoint, speed, and progress topics."""

    def __init__(self) -> None:
        """Declare parameters and wire ROS publishers/subscribers/timer."""
        super().__init__("topo_path_to_waypoint")

        self.declare_parameter("input_path_topic", "/global_path")
        self.declare_parameter("odom_topic", "/fastlio/odom")
        self.declare_parameter("output_waypoint_topic", "/way_point")
        self.declare_parameter("output_speed_topic", "/speed")
        self.declare_parameter("output_progress_topic", "/odom_current_progress")
        self.declare_parameter("topo_meta_topic", "/path_follower/topo_path_meta")
        self.declare_parameter("enable_topology_node_sequence_mode", True)
        self.declare_parameter("topology_node_match_tolerance", 0.25)
        self.declare_parameter("publish_speed", True)
        self.declare_parameter("cruise_speed", 0.6)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("lookahead_distance", 1.2)
        self.declare_parameter("min_waypoint_distance", 0.0)
        self.declare_parameter("goal_tolerance", 0.5)
        self.declare_parameter("goal_distance_mode", "3d")
        self.declare_parameter("projection_distance_mode", "")
        self.declare_parameter("projection_search_distance", 10.0)
        self.declare_parameter("min_path_points", 2)
        self.declare_parameter("start_from_nearest", True)
        self.declare_parameter("loop_mode", "none")
        self.declare_parameter("max_laps", 0)
        self.declare_parameter("loop_pause_sec", 0.0)
        self.declare_parameter("accept_repeated_path", False)
        self.declare_parameter("align_final_orientation", False)
        self.declare_parameter("final_orientation_tolerance", 0.05)
        self.declare_parameter("final_orientation_hold_sec", 0.3)
        self.declare_parameter("final_orientation_timeout_sec", 8.0)
        self.declare_parameter("final_orientation_kp", 1.5)
        self.declare_parameter("final_orientation_min_yaw_rate", 0.0)
        self.declare_parameter("final_orientation_max_yaw_rate", 0.5)
        self.declare_parameter("final_orientation_cmd_vel_topic", "")
        self.declare_parameter("clear_on_goal", True)
        self.declare_parameter("clear_global_path_topic", "")
        self.declare_parameter("reset_waypoint_to_robot_on_goal", False)
        self.declare_parameter("cleanup_stop_topic", "")
        self.declare_parameter("cleanup_stop_value", 2)
        self.declare_parameter("cleanup_cmd_vel_topic", "")
        self.declare_parameter("cleanup_publish_count", 5)

        self.input_path_topic = str(self.get_parameter("input_path_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.output_waypoint_topic = str(self.get_parameter("output_waypoint_topic").value)
        self.output_speed_topic = str(self.get_parameter("output_speed_topic").value)
        self.output_progress_topic = str(
            self.get_parameter("output_progress_topic").value
        ).strip()
        self.topo_meta_topic = str(self.get_parameter("topo_meta_topic").value).strip()
        self.enable_topology_node_sequence_mode = bool(
            self.get_parameter("enable_topology_node_sequence_mode").value
        )
        self.topology_node_match_tolerance = max(
            0.0, float(self.get_parameter("topology_node_match_tolerance").value)
        )
        self.publish_speed = bool(self.get_parameter("publish_speed").value)
        self.cruise_speed = float(self.get_parameter("cruise_speed").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.min_waypoint_distance = max(
            0.0, float(self.get_parameter("min_waypoint_distance").value)
        )
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.goal_distance_mode = self.normalize_distance_mode(
            str(self.get_parameter("goal_distance_mode").value)
        )
        projection_distance_mode = str(
            self.get_parameter("projection_distance_mode").value
        ).strip()
        self.projection_distance_mode = self.normalize_distance_mode(
            projection_distance_mode or self.goal_distance_mode
        )
        self.projection_search_distance = max(
            0.0, float(self.get_parameter("projection_search_distance").value)
        )
        self.min_path_points = int(self.get_parameter("min_path_points").value)
        self.start_from_nearest = bool(self.get_parameter("start_from_nearest").value)
        self.loop_mode = self.normalize_loop_mode(
            str(self.get_parameter("loop_mode").value)
        )
        self.max_laps = max(0, int(self.get_parameter("max_laps").value))
        self.loop_pause_sec = max(0.0, float(self.get_parameter("loop_pause_sec").value))
        self.accept_repeated_path = bool(
            self.get_parameter("accept_repeated_path").value
        )
        self.align_final_orientation = bool(
            self.get_parameter("align_final_orientation").value
        )
        self.final_orientation_tolerance = max(
            0.0, float(self.get_parameter("final_orientation_tolerance").value)
        )
        self.final_orientation_hold_sec = max(
            0.0, float(self.get_parameter("final_orientation_hold_sec").value)
        )
        self.final_orientation_timeout_sec = max(
            0.0, float(self.get_parameter("final_orientation_timeout_sec").value)
        )
        self.final_orientation_kp = max(
            0.0, float(self.get_parameter("final_orientation_kp").value)
        )
        self.final_orientation_min_yaw_rate = max(
            0.0, float(self.get_parameter("final_orientation_min_yaw_rate").value)
        )
        self.final_orientation_max_yaw_rate = max(
            0.0, float(self.get_parameter("final_orientation_max_yaw_rate").value)
        )
        self.final_orientation_min_yaw_rate = min(
            self.final_orientation_min_yaw_rate,
            self.final_orientation_max_yaw_rate,
        )
        self.final_orientation_cmd_vel_topic = str(
            self.get_parameter("final_orientation_cmd_vel_topic").value
        ).strip()
        self.clear_on_goal = bool(self.get_parameter("clear_on_goal").value)
        self.clear_global_path_topic = str(
            self.get_parameter("clear_global_path_topic").value
        ).strip()
        self.reset_waypoint_to_robot_on_goal = bool(
            self.get_parameter("reset_waypoint_to_robot_on_goal").value
        )
        self.cleanup_stop_topic = str(
            self.get_parameter("cleanup_stop_topic").value
        ).strip()
        self.cleanup_stop_value = int(self.get_parameter("cleanup_stop_value").value)
        self.cleanup_cmd_vel_topic = str(
            self.get_parameter("cleanup_cmd_vel_topic").value
        ).strip()
        self.cleanup_publish_count = max(
            1, int(self.get_parameter("cleanup_publish_count").value)
        )

        self.raw_path_points: List[Tuple[float, float, float]] = []
        self.raw_path_yaws: List[Optional[float]] = []
        self.path_points: List[Tuple[float, float, float]] = []
        self.path_yaws: List[Optional[float]] = []
        self.path_segment_lengths: List[float] = []
        self.path_cumulative_lengths: List[float] = [0.0]
        self.path_total_length = 0.0
        self.topo_node_points: List[Tuple[float, float, float]] = []
        self.path_frame_id = "map"
        self.robot_pos: Optional[Tuple[float, float, float]] = None
        self.robot_yaw: Optional[float] = None
        self.current_index = 0
        self.current_segment_index = 0
        self.current_path_s = 0.0
        self.current_projection_point: Optional[Tuple[float, float, float]] = None
        self.path_signature = ""
        self.loop_direction = 1
        self.completed_laps = 0
        self.pause_until: Optional[float] = None
        self.holding_final_goal = False
        self.closed_gap_warned = False
        self.completed_path_signature = ""
        self.cleanup_publishes_remaining = 0
        self.cleanup_clear_path_pending = False
        self.aligning_final_orientation = False
        self.final_orientation_started_at: Optional[float] = None
        self.final_orientation_reached_since: Optional[float] = None
        self.final_orientation_reason = ""
        self.node_waypoint_index: Optional[int] = None
        self.node_waypoint_published = False
        self.active_path_is_topology_nodes = False

        self.path_sub = self.create_subscription(Path, self.input_path_topic, self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.topo_meta_sub = (
            self.create_subscription(
                TopoPathMeta,
                self.topo_meta_topic,
                self.topo_meta_callback,
                10,
            )
            if self.topo_meta_topic
            else None
        )
        self.waypoint_pub = self.create_publisher(PointStamped, self.output_waypoint_topic, 10)
        self.speed_pub = self.create_publisher(Float32, self.output_speed_topic, 10)
        self.progress_pub = (
            self.create_publisher(PointStamped, self.output_progress_topic, 10)
            if self.output_progress_topic
            else None
        )
        clear_path_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.clear_path_pub = (
            self.create_publisher(Path, self.clear_global_path_topic, clear_path_qos)
            if self.clear_global_path_topic
            else None
        )
        self.cleanup_stop_pub = (
            self.create_publisher(Int8, self.cleanup_stop_topic, 10)
            if self.cleanup_stop_topic
            else None
        )
        self.cleanup_cmd_vel_pub = (
            self.create_publisher(Twist, self.cleanup_cmd_vel_topic, 10)
            if self.cleanup_cmd_vel_topic
            else None
        )
        self.final_orientation_cmd_vel_pub = (
            self.create_publisher(
                Twist,
                self.final_orientation_cmd_vel_topic or self.cleanup_cmd_vel_topic,
                10,
            )
            if self.final_orientation_cmd_vel_topic or self.cleanup_cmd_vel_topic
            else None
        )
        self.timer = self.create_timer(max(1.0 / max(self.publish_rate, 1e-3), 1e-3), self.timer_cb)

        self.get_logger().info(
            "TopoPathToWaypoint ready: "
            f"path={self.input_path_topic}, odom={self.odom_topic}, "
            f"waypoint={self.output_waypoint_topic}, speed={self.output_speed_topic}, "
            f"progress={self.output_progress_topic or '<disabled>'}, "
            f"topo_meta={self.topo_meta_topic or '<disabled>'}, "
            f"topology_node_sequence={self.enable_topology_node_sequence_mode}, "
            f"goal_distance_mode={self.goal_distance_mode}, "
            f"projection_distance_mode={self.projection_distance_mode}, "
            f"projection_search_distance={self.projection_search_distance:.2f}, "
            f"lookahead_distance={self.lookahead_distance:.2f}, "
            f"min_waypoint_distance={self.min_waypoint_distance:.2f}, "
            f"loop_mode={self.loop_mode}, max_laps={self.max_laps}, "
            f"loop_pause_sec={self.loop_pause_sec:.2f}, "
            f"accept_repeated_path={self.accept_repeated_path}, "
            f"align_final_orientation={self.align_final_orientation}, "
            f"final_orientation_cmd_vel="
            f"{self.final_orientation_cmd_vel_topic or self.cleanup_cmd_vel_topic or '<disabled>'}, "
            f"clear_on_goal={self.clear_on_goal}, "
            f"clear_global_path={self.clear_global_path_topic or '<disabled>'}, "
            f"cleanup_stop={self.cleanup_stop_topic or '<disabled>'}, "
            f"cleanup_cmd_vel={self.cleanup_cmd_vel_topic or '<disabled>'}"
        )

    def path_callback(self, msg: Path) -> None:
        """Accept a global Path and rebuild the active tracking geometry."""
        if len(msg.poses) == 0:
            had_active_path = bool(self.path_points)
            self.clear_active_path()
            if had_active_path:
                self.start_cleanup_burst("empty global path", publish_clear_path=False)
                self.get_logger().info("Received empty global path; cleared active waypoint task.")
            return

        if len(msg.poses) < self.min_path_points:
            self.clear_active_path()
            self.get_logger().warn(
                f"Received path with {len(msg.poses)} points (< {self.min_path_points}), ignoring."
            )
            return

        raw_points = []
        raw_yaws = []
        for pose_stamped in msg.poses:
            pose = pose_stamped.pose
            raw_points.append(
                (
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                )
            )
            raw_yaws.append(self.quaternion_yaw_or_none(pose.orientation))
        self.path_frame_id = msg.header.frame_id if msg.header.frame_id else "map"

        path_signature = self.compute_path_signature(raw_points, raw_yaws)
        if self.is_completed_path_repeat(path_signature, raw_points):
            self.start_cleanup_burst("completed path repeat", publish_clear_path=True)
            return

        if path_signature == self.path_signature and not self.accept_repeated_path:
            return

        repeated_path = path_signature == self.path_signature
        if path_signature != self.completed_path_signature:
            self.completed_path_signature = ""
        self.raw_path_points = raw_points
        self.raw_path_yaws = raw_yaws
        self.path_signature = path_signature
        self.active_path_is_topology_nodes = self.is_topology_node_path(raw_points)
        self.cancel_cleanup_burst()
        self.cancel_final_orientation_alignment()
        self.reset_loop_state()
        self.rebuild_active_path()

        if self.robot_pos is not None and self.start_from_nearest:
            projection = self.find_closest_projection(
                0, len(self.path_segment_lengths) - 1
            )
            if projection is not None:
                self.set_current_projection(projection)
            else:
                self.reset_path_progress()
        else:
            self.reset_path_progress()
        self.reset_topology_node_waypoint_state()

        self.get_logger().info(
            f"Accepted {'repeated' if repeated_path else 'new'} global path: "
            f"raw_points={len(self.raw_path_points)}, "
            f"active_points={len(self.path_points)}, frame={self.path_frame_id}, "
            f"path_length={self.path_total_length:.2f} m, "
            f"loop_mode={self.loop_mode}, "
            f"tracking={'topology_nodes_once' if self.active_path_is_topology_nodes else 'projected_path'}"
        )

    def topo_meta_callback(self, msg: TopoPathMeta) -> None:
        """Cache topo node metadata for topology-node path detection."""
        self.topo_node_points = [
            (
                float(node.pose.position.x),
                float(node.pose.position.y),
                float(node.pose.position.z),
            )
            for node in msg.nodes
        ]
        if self.raw_path_points:
            was_active = self.active_path_is_topology_nodes
            self.active_path_is_topology_nodes = self.is_topology_node_path(
                self.raw_path_points
            )
            if self.active_path_is_topology_nodes and not was_active:
                self.reset_topology_node_waypoint_state()
                self.get_logger().info(
                    "Detected topology_nodes Path; publishing each node waypoint once."
                )

    def odom_callback(self, msg: Odometry) -> None:
        """Cache robot position and yaw from odometry."""
        p = msg.pose.pose.position
        self.robot_pos = (float(p.x), float(p.y), float(p.z))
        self.robot_yaw = self.quaternion_yaw_or_none(msg.pose.pose.orientation)

    def timer_cb(self) -> None:
        """Project odom to the path and publish progress/waypoint outputs."""
        if self.cleanup_publishes_remaining > 0:
            self.publish_cleanup_outputs()

        if self.aligning_final_orientation:
            self.update_final_orientation_alignment()
            return

        if not self.path_points or self.robot_pos is None:
            return

        if self.pause_until is not None:
            if time.monotonic() < self.pause_until:
                if self.active_path_is_topology_nodes:
                    self.publish_topology_node_waypoint_once(0, 0.0)
                else:
                    self.publish_waypoint(0, 0.0)
                return
            self.pause_until = None

        if self.holding_final_goal:
            if self.clear_on_goal:
                self.finish_after_position_reached("max laps reached")
            elif self.active_path_is_topology_nodes:
                self.publish_topology_node_waypoint_once(
                    len(self.path_points) - 1, 0.0
                )
            else:
                self.publish_waypoint(len(self.path_points) - 1, 0.0)
            return

        path_size = len(self.path_points)
        projection = self.update_current_projection()
        if projection is None:
            return
        self.publish_progress_point(projection.point)

        if self.active_path_is_topology_nodes:
            self.update_topology_node_waypoint()
            return

        if self.complete_loop_lap_if_needed():
            if self.pause_until is not None:
                self.publish_waypoint(0, 0.0)
                return
            if self.holding_final_goal:
                if self.clear_on_goal:
                    self.finish_after_position_reached("max laps reached")
                else:
                    self.publish_waypoint(len(self.path_points) - 1, 0.0)
                return
            path_size = len(self.path_points)

        target_s = self.current_path_s + max(0.0, self.lookahead_distance)
        target_s = self.clamp(target_s, 0.0, self.path_total_length)
        if self.min_waypoint_distance > 0.0:
            target_s = self.advance_s_until_min_xy_distance(
                target_s, self.min_waypoint_distance
            )

        if self.should_finish_single_route_at_s(target_s):
            if self.clear_on_goal:
                self.finish_after_position_reached("goal reached")
                return
            self.publish_waypoint(path_size - 1, 0.0)
            return

        self.publish_waypoint_point(self.point_at_s(target_s), self.cruise_speed)

    def update_topology_node_waypoint(self) -> None:
        """Publish the next topology-node waypoint only when the target changes."""
        if not self.path_points or not self.path_cumulative_lengths:
            return

        if self.node_waypoint_index is None:
            self.node_waypoint_index = self.find_next_topology_node_index()
            self.node_waypoint_published = False

        while self.node_waypoint_index is not None:
            target_index = self.node_waypoint_index
            if not self.has_reached_topology_node_waypoint(target_index):
                self.publish_topology_node_waypoint_once(
                    target_index, self.cruise_speed
                )
                return

            if target_index >= len(self.path_points) - 1:
                if self.loop_mode == "none":
                    if self.clear_on_goal:
                        self.finish_after_position_reached("goal reached")
                    else:
                        self.holding_final_goal = True
                        self.publish_topology_node_waypoint_once(target_index, 0.0)
                    return

                self.complete_topology_node_lap()
                if self.pause_until is not None:
                    self.publish_topology_node_waypoint_once(0, 0.0)
                    return
                self.node_waypoint_index = self.find_next_topology_node_index()
                self.node_waypoint_published = False
                continue

            self.node_waypoint_index = target_index + 1
            self.node_waypoint_published = False

    def find_next_topology_node_index(self) -> Optional[int]:
        """Find the nearest active path node ahead of current projection progress."""
        if not self.path_points or not self.path_cumulative_lengths:
            return None

        reach_tolerance = max(0.0, self.goal_tolerance)
        for index, node_s in enumerate(self.path_cumulative_lengths):
            if node_s + reach_tolerance >= self.current_path_s:
                return index
        return len(self.path_points) - 1

    def has_reached_topology_node_waypoint(self, index: int) -> bool:
        """Check whether progress projection has reached the current node waypoint."""
        if not self.path_points or not self.path_cumulative_lengths:
            return False

        index = max(0, min(index, len(self.path_points) - 1))
        reach_tolerance = max(0.0, self.goal_tolerance)
        target_s = self.path_cumulative_lengths[index]
        if self.current_path_s >= target_s - reach_tolerance:
            return True

        if self.current_projection_point is None:
            return False
        return (
            self.goal_distance(self.current_projection_point, self.path_points[index])
            <= reach_tolerance
        )

    def publish_topology_node_waypoint_once(self, index: int, speed: float) -> None:
        """Publish a topology-node waypoint only once for the active node target."""
        if not self.path_points:
            return

        index = max(0, min(index, len(self.path_points) - 1))
        if self.node_waypoint_index != index:
            self.node_waypoint_index = index
            self.node_waypoint_published = False
        if self.node_waypoint_published:
            return

        self.publish_waypoint(index, speed)
        self.node_waypoint_published = True

    def reset_topology_node_waypoint_state(self) -> None:
        """Forget the current topology-node waypoint publication state."""
        self.node_waypoint_index = None
        self.node_waypoint_published = False

    def complete_topology_node_lap(self) -> None:
        """Advance loop state after the projected progress reaches the final node."""
        self.completed_laps += 1
        if self.max_laps > 0 and self.completed_laps >= self.max_laps:
            self.current_index = len(self.path_points) - 1
            self.current_segment_index = max(0, len(self.path_segment_lengths) - 1)
            self.current_path_s = self.path_total_length
            self.holding_final_goal = True
            self.get_logger().info(
                f"Loop complete: reached configured max_laps={self.max_laps}; holding final goal."
            )
            return

        if self.loop_mode == "pingpong":
            self.loop_direction *= -1

        self.rebuild_active_path()
        self.reset_path_progress()

        if self.loop_pause_sec > 0.0:
            self.pause_until = time.monotonic() + self.loop_pause_sec

        self.get_logger().info(
            f"Topology-node loop lap {self.completed_laps} complete; continuing with "
            f"direction={'forward' if self.loop_direction > 0 else 'reverse'}."
        )

    def should_finish_single_route_at_s(self, waypoint_s: float) -> bool:
        """Return true when a non-loop route should finish at this arc length."""
        if self.loop_mode != "none":
            return False
        if waypoint_s < self.path_total_length - 1e-6:
            return False
        return self.has_reached_final_goal()

    def has_reached_final_goal(self) -> bool:
        """Check whether the robot is within final-goal tolerance."""
        if not self.path_points or self.robot_pos is None:
            return False

        return self.goal_distance(self.robot_pos, self.path_points[-1]) <= self.goal_tolerance

    def complete_loop_lap_if_needed(self) -> bool:
        """Advance closed/pingpong loop state after reaching the final point."""
        if (
            self.loop_mode == "none"
            or not self.path_points
            or not self.path_segment_lengths
            or self.robot_pos is None
        ):
            return False

        if self.current_segment_index < max(0, len(self.path_segment_lengths) - 1):
            return False

        if not self.has_reached_final_goal():
            return False

        self.completed_laps += 1
        if self.max_laps > 0 and self.completed_laps >= self.max_laps:
            self.current_index = len(self.path_points) - 1
            self.current_segment_index = max(0, len(self.path_segment_lengths) - 1)
            self.current_path_s = self.path_total_length
            self.holding_final_goal = True
            self.get_logger().info(
                f"Loop complete: reached configured max_laps={self.max_laps}; holding final goal."
            )
            return True

        if self.loop_mode == "pingpong":
            self.loop_direction *= -1

        self.rebuild_active_path()
        self.reset_path_progress()

        if self.loop_pause_sec > 0.0:
            self.pause_until = time.monotonic() + self.loop_pause_sec

        self.get_logger().info(
            f"Loop lap {self.completed_laps} complete; continuing with "
            f"direction={'forward' if self.loop_direction > 0 else 'reverse'}."
        )
        return True

    def rebuild_active_path(self) -> None:
        """Apply loop direction/closure and rebuild active path geometry."""
        if self.loop_mode == "pingpong" and self.loop_direction < 0:
            self.path_points = list(reversed(self.raw_path_points))
            self.path_yaws = list(reversed(self.raw_path_yaws))
        else:
            self.path_points = list(self.raw_path_points)
            self.path_yaws = list(self.raw_path_yaws)
            if self.loop_mode == "closed" and len(self.path_points) >= 2:
                start = self.path_points[0]
                end = self.path_points[-1]
                if self.dist3(start, end) > 1e-6:
                    if not self.closed_gap_warned:
                        gap = self.dist3(start, end)
                        self.get_logger().warn(
                            "closed loop path does not end at the start point; "
                            f"adding a direct closing segment of {gap:.2f} m. "
                            "Prefer providing an explicit return edge in the topo JSON."
                        )
                        self.closed_gap_warned = True

                    self.path_points.append(start)
                    self.path_yaws.append(self.path_yaws[0] if self.path_yaws else None)

        self.rebuild_path_geometry()

    def rebuild_path_geometry(self) -> None:
        """Recompute segment lengths and cumulative path arc lengths."""
        self.path_segment_lengths = []
        self.path_cumulative_lengths = [0.0]
        total = 0.0
        for start, end in zip(self.path_points, self.path_points[1:]):
            length = self.dist3(start, end)
            self.path_segment_lengths.append(length)
            total += length
            self.path_cumulative_lengths.append(total)
        self.path_total_length = total

    def reset_path_progress(self) -> None:
        """Reset current projection/progress to the path start."""
        self.current_index = 0
        self.current_segment_index = 0
        self.current_path_s = 0.0
        self.current_projection_point = self.path_points[0] if self.path_points else None
        self.reset_topology_node_waypoint_state()

    def reset_loop_state(self) -> None:
        """Reset loop counters, pause state, and hold flags."""
        self.loop_direction = 1
        self.completed_laps = 0
        self.pause_until = None
        self.holding_final_goal = False
        self.closed_gap_warned = False

    def complete_current_task(self, reason: str) -> None:
        """Stop outputs, mark the path complete, and clear active tracking."""
        if not self.path_points:
            return

        if self.publish_speed:
            self.publish_speed_value(0.0)
        self.completed_path_signature = self.path_signature
        self.start_cleanup_burst(reason, publish_clear_path=True)
        self.clear_active_path()
        self.get_logger().info(
            f"Global path task complete ({reason}); cleared active waypoint task."
        )

    def clear_active_path(self) -> None:
        """Drop the active path and related progress state."""
        self.raw_path_points = []
        self.raw_path_yaws = []
        self.path_points = []
        self.path_yaws = []
        self.path_segment_lengths = []
        self.path_cumulative_lengths = [0.0]
        self.path_total_length = 0.0
        self.active_path_is_topology_nodes = False
        self.reset_path_progress()
        self.path_signature = ""
        self.cancel_final_orientation_alignment()
        self.reset_loop_state()

    def finish_after_position_reached(self, reason: str) -> None:
        """Start final yaw alignment or complete the task immediately."""
        if self.start_final_orientation_alignment(reason):
            return
        self.complete_current_task(reason)

    def start_final_orientation_alignment(self, reason: str) -> bool:
        """Begin final in-place yaw alignment if configured and possible."""
        if not self.align_final_orientation or not self.path_yaws:
            return False

        target_yaw = self.path_yaws[-1]
        if target_yaw is None:
            return False

        if self.robot_yaw is None:
            self.get_logger().warn(
                "Skipping final orientation alignment because odometry orientation is invalid."
            )
            return False

        self.aligning_final_orientation = True
        self.final_orientation_started_at = time.monotonic()
        self.final_orientation_reached_since = None
        self.final_orientation_reason = reason
        self.publish_stop_outputs()
        if self.publish_speed:
            self.publish_speed_value(0.0)
        self.get_logger().info(
            "Final goal position reached; aligning yaw to "
            f"{target_yaw:.3f} rad before completing task."
        )
        return True

    def cancel_final_orientation_alignment(self) -> None:
        """Clear final-yaw alignment state."""
        self.aligning_final_orientation = False
        self.final_orientation_started_at = None
        self.final_orientation_reached_since = None
        self.final_orientation_reason = ""

    def update_final_orientation_alignment(self) -> None:
        """Drive final yaw alignment until held, timed out, or unavailable."""
        if not self.path_points:
            self.cancel_final_orientation_alignment()
            return

        target_yaw = self.path_yaws[-1] if self.path_yaws else None
        if target_yaw is None or self.robot_yaw is None:
            reason = self.final_orientation_reason or "goal reached"
            self.cancel_final_orientation_alignment()
            self.complete_current_task(
                f"{reason}, final orientation unavailable"
            )
            return

        now = time.monotonic()
        yaw_error = self.wrap_pi(target_yaw - self.robot_yaw)
        abs_error = math.fabs(yaw_error)
        if abs_error <= self.final_orientation_tolerance:
            self.publish_final_orientation_cmd(0.0)
            if self.final_orientation_reached_since is None:
                self.final_orientation_reached_since = now
                return
            if now - self.final_orientation_reached_since >= self.final_orientation_hold_sec:
                reason = self.final_orientation_reason or "goal reached"
                self.cancel_final_orientation_alignment()
                self.complete_current_task(f"{reason}, final orientation aligned")
            return

        self.final_orientation_reached_since = None
        if (
            self.final_orientation_timeout_sec > 0.0
            and self.final_orientation_started_at is not None
            and now - self.final_orientation_started_at >= self.final_orientation_timeout_sec
        ):
            self.get_logger().warn(
                "Final orientation alignment timed out with yaw_error="
                f"{yaw_error:.3f} rad."
            )
            reason = self.final_orientation_reason or "goal reached"
            self.cancel_final_orientation_alignment()
            self.complete_current_task(f"{reason}, final orientation timeout")
            return

        yaw_rate = self.final_orientation_kp * yaw_error
        yaw_rate = self.clamp(
            yaw_rate,
            -self.final_orientation_max_yaw_rate,
            self.final_orientation_max_yaw_rate,
        )
        if (
            self.final_orientation_min_yaw_rate > 0.0
            and math.fabs(yaw_rate) < self.final_orientation_min_yaw_rate
        ):
            yaw_rate = math.copysign(self.final_orientation_min_yaw_rate, yaw_error)
        self.publish_final_orientation_cmd(yaw_rate)

    def publish_final_orientation_cmd(self, yaw_rate: float) -> None:
        """Publish a yaw-rate command on the final-align channel."""
        if self.publish_speed:
            self.publish_speed_value(0.0)
        self.publish_stop_outputs()
        if self.final_orientation_cmd_vel_pub is None:
            return
        cmd = Twist()
        cmd.angular.z = float(yaw_rate)
        self.final_orientation_cmd_vel_pub.publish(cmd)

    def publish_stop_outputs(self) -> None:
        """Publish the CMU stop command if that output is enabled."""
        if self.cleanup_stop_pub is not None:
            self.cleanup_stop_pub.publish(Int8(data=self.cleanup_stop_value))

    def is_completed_path_repeat(
        self, path_signature: str, points: List[Tuple[float, float, float]]
    ) -> bool:
        """Detect a repeated path that was already completed near its goal."""
        if path_signature != self.completed_path_signature:
            return False
        if self.robot_pos is None:
            return True
        if not points:
            return True
        return self.goal_distance(self.robot_pos, points[-1]) <= self.goal_tolerance

    def publish_waypoint(self, index: int, speed: float) -> None:
        """Publish a discrete active-path point by index as /way_point."""
        if not self.path_points:
            return

        index = max(0, min(index, len(self.path_points) - 1))
        self.publish_waypoint_point(self.path_points[index], speed)

    def publish_waypoint_point(self, point: Tuple[float, float, float], speed: float) -> None:
        """Publish an arbitrary target point and optional speed hint."""
        wp = PointStamped()
        wp.header.stamp = self.get_clock().now().to_msg()
        wp.header.frame_id = self.path_frame_id
        wp.point.x = point[0]
        wp.point.y = point[1]
        wp.point.z = point[2]
        self.waypoint_pub.publish(wp)

        if self.publish_speed:
            self.publish_speed_value(speed)

    def publish_progress_point(self, point: Tuple[float, float, float]) -> None:
        """Publish the odom projection point for progress/debug display."""
        if self.progress_pub is None:
            return

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id
        msg.point.x = point[0]
        msg.point.y = point[1]
        msg.point.z = point[2]
        self.progress_pub.publish(msg)

    def publish_speed_value(self, speed: float) -> None:
        """Publish a scalar speed hint."""
        self.speed_pub.publish(Float32(data=float(speed)))

    def start_cleanup_burst(self, reason: str, publish_clear_path: bool) -> None:
        """Schedule repeated stop/speed/path-clear outputs after completion."""
        self.cleanup_publishes_remaining = max(
            self.cleanup_publishes_remaining, self.cleanup_publish_count
        )
        self.cleanup_clear_path_pending = self.cleanup_clear_path_pending or publish_clear_path
        self.publish_cleanup_outputs()
        self.get_logger().debug(f"Started navigation cleanup burst: {reason}.")

    def cancel_cleanup_burst(self) -> None:
        """Cancel pending cleanup outputs."""
        self.cleanup_publishes_remaining = 0
        self.cleanup_clear_path_pending = False

    def publish_cleanup_outputs(self) -> None:
        """Emit one cleanup tick of stop, speed, and optional path-clear outputs."""
        if self.cleanup_publishes_remaining <= 0:
            return

        if self.reset_waypoint_to_robot_on_goal and self.robot_pos is not None:
            self.publish_waypoint_point(self.robot_pos, 0.0)
        elif self.publish_speed:
            self.publish_speed_value(0.0)

        if self.cleanup_stop_pub is not None:
            self.cleanup_stop_pub.publish(Int8(data=self.cleanup_stop_value))

        if self.cleanup_cmd_vel_pub is not None:
            self.cleanup_cmd_vel_pub.publish(Twist())

        if self.cleanup_clear_path_pending:
            self.publish_empty_global_path()
            self.cleanup_clear_path_pending = False

        self.cleanup_publishes_remaining -= 1

    def publish_empty_global_path(self) -> None:
        """Publish an empty Path to clear downstream global-path state."""
        if self.clear_path_pub is None:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id
        self.clear_path_pub.publish(msg)

    def is_topology_node_path(
        self, points: List[Tuple[float, float, float]]
    ) -> bool:
        """Check whether a Path matches the topo metadata node sequence."""
        if not self.enable_topology_node_sequence_mode:
            return False
        if len(self.topo_node_points) < 2 or len(points) < 2:
            return False

        node_count = len(self.topo_node_points)
        if len(points) == node_count:
            compare_points = points
        elif (
            len(points) == node_count + 1
            and self.dist3(points[-1], self.topo_node_points[0])
            <= self.topology_node_match_tolerance
        ):
            compare_points = points[:-1]
        else:
            return False

        return all(
            self.dist3(point, topo_point) <= self.topology_node_match_tolerance
            for point, topo_point in zip(compare_points, self.topo_node_points)
        )

    def update_current_projection(self) -> Optional[PathProjection]:
        """Project the robot inside the arc-length search window and update progress."""
        if self.robot_pos is None or not self.path_segment_lengths:
            return None

        search_start, search_end = self.segment_range_for_s_window(
            self.current_path_s, self.projection_search_distance
        )
        projection = self.find_closest_projection(search_start, search_end)
        if projection is None:
            return None
        self.set_current_projection(projection)
        return projection

    def segment_range_for_s_window(
        self, center_s: float, window_distance: float
    ) -> Tuple[int, int]:
        """Convert an arc-length window into inclusive segment indices."""
        if not self.path_segment_lengths:
            return 0, 0

        segment_count = len(self.path_segment_lengths)
        if window_distance <= 0.0:
            return 0, segment_count - 1

        low_s = self.clamp(center_s - window_distance, 0.0, self.path_total_length)
        high_s = self.clamp(center_s + window_distance, 0.0, self.path_total_length)
        if low_s > high_s:
            low_s, high_s = high_s, low_s

        start_segment = 0
        end_segment = segment_count - 1
        for idx in range(segment_count):
            if self.path_cumulative_lengths[idx + 1] >= low_s:
                start_segment = idx
                break
        for idx in range(start_segment, segment_count):
            if self.path_cumulative_lengths[idx] > high_s:
                end_segment = max(start_segment, idx - 1)
                break
        return start_segment, end_segment

    def find_closest_projection(
        self, start_segment: int, end_segment: int
    ) -> Optional[PathProjection]:
        """Find the nearest robot projection within a segment-index range."""
        if self.robot_pos is None or not self.path_segment_lengths:
            return None

        segment_count = len(self.path_segment_lengths)
        start_segment = max(0, min(start_segment, segment_count - 1))
        end_segment = max(0, min(end_segment, segment_count - 1))
        if start_segment > end_segment:
            start_segment, end_segment = end_segment, start_segment

        best: Optional[PathProjection] = None
        best_progress_delta = float("inf")
        for segment_index in range(start_segment, end_segment + 1):
            projection = self.project_robot_to_segment(segment_index)
            progress_delta = math.fabs(projection.path_s - self.current_path_s)
            if (
                best is None
                or projection.distance < best.distance - 1e-6
                or (
                    math.fabs(projection.distance - best.distance) <= 1e-6
                    and progress_delta < best_progress_delta
                )
            ):
                best = projection
                best_progress_delta = progress_delta
        return best

    def set_current_projection(self, projection: PathProjection) -> None:
        """Store current segment, arc length, projection point, and index hint."""
        self.current_segment_index = projection.segment_index
        self.current_path_s = projection.path_s
        self.current_projection_point = projection.point
        next_index = 1 if projection.segment_t >= 0.999 else 0
        self.current_index = max(
            0,
            min(projection.segment_index + next_index, len(self.path_points) - 1),
        )

    def project_robot_to_segment(self, segment_index: int) -> PathProjection:
        """Project the robot position onto one active path segment."""
        start = self.path_points[segment_index]
        end = self.path_points[segment_index + 1]
        t = self.segment_projection_t(self.robot_pos, start, end)  # type: ignore[arg-type]
        point = self.interpolate_point(start, end, t)
        segment_length = self.path_segment_lengths[segment_index]
        path_s = self.path_cumulative_lengths[segment_index] + t * segment_length
        return PathProjection(
            segment_index=segment_index,
            segment_t=t,
            path_s=path_s,
            point=point,
            distance=self.projection_distance(self.robot_pos, point),  # type: ignore[arg-type]
        )

    def segment_projection_t(
        self,
        point: Tuple[float, float, float],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> float:
        """Compute the clamped interpolation ratio for a point on a segment."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        use_z = self.projection_distance_mode == "3d" or dx * dx + dy * dy <= 1e-12
        denom = dx * dx + dy * dy + (dz * dz if use_z else 0.0)
        if denom <= 1e-12:
            return 0.0

        dot = (
            (point[0] - start[0]) * dx
            + (point[1] - start[1]) * dy
            + ((point[2] - start[2]) * dz if use_z else 0.0)
        )
        return self.clamp(dot / denom, 0.0, 1.0)

    def projection_distance(
        self, a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> float:
        """Measure projection error using the configured 3D/XY mode."""
        if self.projection_distance_mode == "xy":
            return self.dist_xy(a, b)
        return self.dist3(a, b)

    def point_at_s(self, path_s: float) -> Tuple[float, float, float]:
        """Interpolate the active path at an arc-length coordinate."""
        if not self.path_points:
            return (0.0, 0.0, 0.0)
        if not self.path_segment_lengths or self.path_total_length <= 1e-9:
            return self.path_points[-1]

        path_s = self.clamp(path_s, 0.0, self.path_total_length)
        if path_s <= 0.0:
            return self.path_points[0]
        if path_s >= self.path_total_length:
            return self.path_points[-1]

        segment_index, segment_t = self.locate_s(path_s)
        return self.interpolate_point(
            self.path_points[segment_index],
            self.path_points[segment_index + 1],
            segment_t,
        )

    def locate_s(self, path_s: float) -> Tuple[int, float]:
        """Find segment index and interpolation ratio for an arc length."""
        if not self.path_segment_lengths:
            return 0, 0.0

        path_s = self.clamp(path_s, 0.0, self.path_total_length)
        for segment_index, segment_length in enumerate(self.path_segment_lengths):
            segment_start_s = self.path_cumulative_lengths[segment_index]
            segment_end_s = self.path_cumulative_lengths[segment_index + 1]
            if path_s <= segment_end_s or segment_index == len(self.path_segment_lengths) - 1:
                if segment_length <= 1e-9:
                    return segment_index, 0.0
                return segment_index, self.clamp(
                    (path_s - segment_start_s) / segment_length, 0.0, 1.0
                )
        return len(self.path_segment_lengths) - 1, 1.0

    def advance_s_until_min_xy_distance(
        self, start_s: float, min_distance: float
    ) -> float:
        """Advance arc length until the target is far enough from the robot in XY."""
        if self.robot_pos is None or not self.path_segment_lengths:
            return start_s

        start_s = self.clamp(start_s, 0.0, self.path_total_length)
        if self.dist_xy(self.robot_pos, self.point_at_s(start_s)) >= min_distance:
            return start_s

        segment_index, segment_t = self.locate_s(start_s)
        for idx in range(segment_index, len(self.path_segment_lengths)):
            begin_t = segment_t if idx == segment_index else 0.0
            candidate_t = self.first_segment_t_at_xy_distance(
                idx, begin_t, min_distance
            )
            if candidate_t is not None:
                return self.path_cumulative_lengths[idx] + (
                    candidate_t * self.path_segment_lengths[idx]
                )
        return self.path_total_length

    def first_segment_t_at_xy_distance(
        self, segment_index: int, begin_t: float, distance: float
    ) -> Optional[float]:
        """Find the first point on a segment at the requested XY robot distance."""
        if self.robot_pos is None:
            return None

        start = self.path_points[segment_index]
        end = self.path_points[segment_index + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        a = dx * dx + dy * dy
        begin_t = self.clamp(begin_t, 0.0, 1.0)

        begin_point = self.interpolate_point(start, end, begin_t)
        if self.dist_xy(self.robot_pos, begin_point) >= distance:
            return begin_t
        end_point = self.interpolate_point(start, end, 1.0)
        if self.dist_xy(self.robot_pos, end_point) < distance:
            return None
        if a <= 1e-12:
            return 1.0

        x0 = start[0] - self.robot_pos[0]
        y0 = start[1] - self.robot_pos[1]
        b = 2.0 * (x0 * dx + y0 * dy)
        c = x0 * x0 + y0 * y0 - distance * distance
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return 1.0

        sqrt_discriminant = math.sqrt(discriminant)
        roots = sorted(
            (
                (-b - sqrt_discriminant) / (2.0 * a),
                (-b + sqrt_discriminant) / (2.0 * a),
            )
        )
        for root in roots:
            if root < begin_t - 1e-9 or root > 1.0 + 1e-9:
                continue
            candidate_t = self.clamp(root, begin_t, 1.0)
            candidate_point = self.interpolate_point(start, end, candidate_t)
            if self.dist_xy(self.robot_pos, candidate_point) >= distance - 1e-6:
                return candidate_t
        return 1.0

    @staticmethod
    def interpolate_point(
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        t: float,
    ) -> Tuple[float, float, float]:
        """Linearly interpolate between two 3D points."""
        return (
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t,
        )

    @staticmethod
    def compute_path_signature(
        points: List[Tuple[float, float, float]],
        yaws: Optional[List[Optional[float]]] = None,
    ) -> str:
        """Build a compact path signature used to detect repeats."""
        point_text = ";".join(f"{x:.3f},{y:.3f},{z:.3f}" for x, y, z in points)
        final_yaw = yaws[-1] if yaws else None
        if final_yaw is None:
            return point_text
        return f"{point_text}|final_yaw={final_yaw:.3f}"

    @staticmethod
    def normalize_loop_mode(value: str) -> str:
        """Normalize loop-mode aliases."""
        mode = value.strip().lower()
        if mode in ("", "none", "off", "false", "0", "no"):
            return "none"
        if mode in ("closed", "circle", "circular", "loop"):
            return "closed"
        if mode in ("pingpong", "ping_pong", "backforth", "back_and_forth"):
            return "pingpong"
        raise ValueError(
            f"Unsupported loop_mode '{value}'. Use 'none', 'closed', or 'pingpong'."
        )

    @staticmethod
    def normalize_distance_mode(value: str) -> str:
        """Normalize distance-mode aliases."""
        mode = value.strip().lower()
        if mode in ("", "3d", "xyz"):
            return "3d"
        if mode in ("2d", "xy", "planar"):
            return "xy"
        raise ValueError(
            f"Unsupported goal_distance_mode '{value}'. Use '3d' or 'xy'."
        )

    def goal_distance(
        self, a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> float:
        """Measure goal distance using the configured 3D/XY mode."""
        if self.goal_distance_mode == "xy":
            return self.dist_xy(a, b)
        return self.dist3(a, b)

    @staticmethod
    def dist_xy(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Return planar distance between two 3D tuples."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """Return Euclidean 3D distance between two points."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def quaternion_yaw_or_none(quat) -> Optional[float]:
        """Extract yaw from a quaternion, or None if it is invalid."""
        qx = float(quat.x)
        qy = float(quat.y)
        qz = float(quat.z)
        qw = float(quat.w)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            return None
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    @staticmethod
    def wrap_pi(value: float) -> float:
        """Wrap an angle to [-pi, pi]."""
        return math.atan2(math.sin(value), math.cos(value))

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        """Clamp a value into inclusive bounds."""
        if low > high:
            low, high = high, low
        return max(low, min(high, value))


def main(args=None) -> None:
    """Start the ROS node."""
    rclpy.init(args=args)
    node = TopoPathToWaypoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
