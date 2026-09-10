"""Standalone global Path -> waypoint bridge bringup.

Starts galileo_nav/topo_path_to_waypoint as the canonical path-to-waypoint
adapter:

  /global_path + odom -> /way_point, /speed, /odom_current_progress

Parameters default to config/path_to_waypoint.yaml. local_cmu and nav_main
include this launch for the CMU waypoint bridge, but it can also be run directly
for bag/offline waypoint generation checks.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


TRUE_VALUES = "['true', '1', 'yes', 'on']"
BOOL_TRUE_VALUES = ("true", "1", "yes", "on")
KINEMATIC_MODEL_ALIASES = {
    "": "",
    "default": "original",
    "original": "original",
    "legacy": "original",
    "omni": "original",
    "holonomic": "original",
    "vehicle": "vehicle",
    "car": "vehicle",
    "ackermann": "vehicle",
}
LINEAR_SPEED_NODE_PARAMS = {
    "path_tracking": ("cruise_speed",),
    "path_to_waypoint": ("cruise_speed",),
    "topo_path_to_waypoint": ("cruise_speed",),
    "path_to_cmu_waypoint": ("cruise_speed",),
}
MODEL_SPEED_NODE_PARAMS = {
    "path_tracking": (
        ("final_orientation_min_yaw_rate", "final_orientation_min_yaw_rate"),
        ("final_orientation_max_yaw_rate", "final_orientation_max_yaw_rate"),
    ),
    "path_to_waypoint": (
        ("final_orientation_min_yaw_rate", "final_orientation_min_yaw_rate"),
        ("final_orientation_max_yaw_rate", "final_orientation_max_yaw_rate"),
    ),
    "topo_path_to_waypoint": (
        ("final_orientation_min_yaw_rate", "final_orientation_min_yaw_rate"),
        ("final_orientation_max_yaw_rate", "final_orientation_max_yaw_rate"),
    ),
    "path_to_cmu_waypoint": (
        ("final_orientation_min_yaw_rate", "final_orientation_min_yaw_rate"),
        ("final_orientation_max_yaw_rate", "final_orientation_max_yaw_rate"),
    ),
}


def _bool_condition(name):
    return IfCondition(
        PythonExpression(["'", LaunchConfiguration(name), "'.lower() in ", TRUE_VALUES])
    )


def _as_bool(value):
    return str(value).strip().lower() in BOOL_TRUE_VALUES


def _optional_override(context, launch_name, param_name, value_type):
    raw_value = LaunchConfiguration(launch_name).perform(context).strip()
    if raw_value == "":
        return {}

    if value_type is bool:
        return {param_name: _as_bool(raw_value)}
    if value_type is int:
        return {param_name: int(raw_value)}
    if value_type is float:
        return {param_name: float(raw_value)}
    return {param_name: raw_value}


def _load_config_params(config_file):
    with open(os.path.expanduser(config_file), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _node_params_from_mapping(mapping, node_name):
    node_data = mapping.get(node_name, {})
    if not isinstance(node_data, dict):
        return {}
    params = node_data.get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


def _node_params(config_data, node_name, section_name=None):
    if section_name:
        section_data = config_data.get(section_name, {})
        if isinstance(section_data, dict):
            params = _node_params_from_mapping(section_data, node_name)
            if params:
                return params
    return _node_params_from_mapping(config_data, node_name)


def _model_value(model_data, key):
    if key not in model_data:
        return None
    value = model_data[key]
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _normalize_kinematic_model(value):
    key = str(value or "").strip().lower()
    if key not in KINEMATIC_MODEL_ALIASES:
        available = ", ".join(sorted(k for k in KINEMATIC_MODEL_ALIASES if k))
        raise RuntimeError(
            f"Unsupported kinematic_model '{value}'. Available aliases: {available}."
        )
    return KINEMATIC_MODEL_ALIASES[key]


def _configured_kinematic_model(config_data, section):
    section_data = config_data.get(section, {})
    if isinstance(section_data, dict):
        section_value = section_data.get("kinematic_model", "")
        if str(section_value).strip():
            return _normalize_kinematic_model(section_value)
    return _normalize_kinematic_model(config_data.get("kinematic_model", "original"))


def _selected_kinematic_model(context, config_data, section):
    launch_value = LaunchConfiguration("kinematic_model").perform(context).strip()
    if launch_value:
        return _normalize_kinematic_model(launch_value)
    return _configured_kinematic_model(config_data, section)


def _kinematic_node_params(config_data, kinematic_model, node_name):
    model_name = _normalize_kinematic_model(kinematic_model)
    if not model_name:
        return {}
    model_data = config_data.get("kinematic_models", {}).get(model_name, {})
    if not isinstance(model_data, dict):
        return {}

    params = {}
    linear_speed = _model_value(model_data, "linear_speed")
    if linear_speed is not None:
        for param_name in LINEAR_SPEED_NODE_PARAMS.get(node_name, ()):
            params[param_name] = linear_speed

    for model_key, param_name in MODEL_SPEED_NODE_PARAMS.get(node_name, ()):
        value = _model_value(model_data, model_key)
        if value is not None:
            params[param_name] = value

    params.update(_node_params_from_mapping(model_data, node_name))
    return params


def _section_node_params(config_data, section, *node_names, kinematic_model=""):
    params = {}
    for node_name in node_names:
        params.update(_node_params(config_data, node_name, "common"))
    for node_name in node_names:
        params.update(_node_params(config_data, node_name, section))
    for node_name in node_names:
        params.update(_kinematic_node_params(config_data, kinematic_model, node_name))
    for node_name in node_names:
        params.update(_node_params(config_data, node_name))
    return params


def _section_path_tracking_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data,
        section,
        "path_tracking",
        "path_to_waypoint",
        "topo_path_to_waypoint",
        "path_to_cmu_waypoint",
        kinematic_model=kinematic_model,
    )


def _default_config_file(pkg_galileo_nav):
    candidates = [
        os.path.join(pkg_galileo_nav, "config", "path_to_waypoint.yaml"),
        os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "config",
                "path_to_waypoint.yaml",
            )
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _build_path_to_waypoint(context, *args, **kwargs):
    config_data = _load_config_params(LaunchConfiguration("config_file").perform(context))
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    section = "sim" if mode == "sim" else "real"
    kinematic_model = _selected_kinematic_model(context, config_data, section)

    params = _section_path_tracking_params(config_data, section, kinematic_model)
    final_goal_tolerance = (
        LaunchConfiguration("final_goal_tolerance").perform(context).strip()
    )

    launch_overrides = {}
    if final_goal_tolerance:
        launch_overrides["goal_tolerance"] = float(final_goal_tolerance)
    launch_overrides.update(
        _optional_override(context, "bridge_publish_rate", "publish_rate", float)
    )
    launch_overrides.update(
        _optional_override(context, "bridge_lookahead", "lookahead_distance", float)
    )
    launch_overrides.update(
        _optional_override(
            context,
            "bridge_min_waypoint_distance",
            "min_waypoint_distance",
            float,
        )
    )
    launch_overrides.update(
        _optional_override(context, "bridge_goal_tolerance", "goal_tolerance", float)
    )
    launch_overrides.update(
        _optional_override(
            context,
            "bridge_projection_search_distance",
            "projection_search_distance",
            float,
        )
    )
    launch_overrides.update(
        _optional_override(context, "bridge_loop_mode", "loop_mode", str)
    )
    launch_overrides.update(
        _optional_override(context, "bridge_max_laps", "max_laps", int)
    )
    launch_overrides.update(
        _optional_override(context, "bridge_loop_pause_sec", "loop_pause_sec", float)
    )
    launch_overrides.update(
        _optional_override(
            context, "bridge_accept_repeated_path", "accept_repeated_path", bool
        )
    )
    cruise_speed_override = LaunchConfiguration("cruise_speed").perform(context).strip()
    if cruise_speed_override:
        launch_overrides["cruise_speed"] = float(cruise_speed_override)

    cleanup_stop_value = int(
        LaunchConfiguration("cleanup_stop_value").perform(context).strip() or "2"
    )

    return [
        Node(
            package="galileo_nav",
            executable="topo_path_to_waypoint",
            name=LaunchConfiguration("node_name"),
            output="screen",
            parameters=[
                params,
                {
                    "input_path_topic": LaunchConfiguration("input_path_topic"),
                    "odom_topic": LaunchConfiguration("odom_topic"),
                    "output_waypoint_topic": LaunchConfiguration(
                        "output_waypoint_topic"
                    ),
                    "output_speed_topic": LaunchConfiguration("output_speed_topic"),
                    "output_progress_topic": LaunchConfiguration(
                        "output_progress_topic"
                    ),
                    "clear_global_path_topic": LaunchConfiguration(
                        "clear_global_path_topic"
                    ),
                    "cleanup_stop_topic": LaunchConfiguration("cleanup_stop_topic"),
                    "cleanup_stop_value": cleanup_stop_value,
                    "cleanup_cmd_vel_topic": LaunchConfiguration(
                        "cleanup_cmd_vel_topic"
                    ),
                    "final_orientation_cmd_vel_topic": LaunchConfiguration(
                        "final_orientation_cmd_vel_topic"
                    ),
                },
                launch_overrides,
            ],
            condition=_bool_condition("enable_path_to_waypoint"),
        )
    ]


def generate_launch_description():
    pkg_galileo_nav = get_package_share_directory("galileo_nav")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=_default_config_file(pkg_galileo_nav),
                description="Path to path_to_waypoint.yaml.",
            ),
            DeclareLaunchArgument("mode", default_value="real"),
            DeclareLaunchArgument(
                "kinematic_model",
                default_value="",
                description="Kinematic model profile: original/vehicle; empty uses config.",
            ),
            DeclareLaunchArgument("enable_path_to_waypoint", default_value="true"),
            DeclareLaunchArgument("node_name", default_value="path_to_cmu_waypoint"),
            DeclareLaunchArgument("input_path_topic", default_value="/global_path"),
            DeclareLaunchArgument("odom_topic", default_value="/state_estimation"),
            DeclareLaunchArgument("output_waypoint_topic", default_value="/way_point"),
            DeclareLaunchArgument("output_speed_topic", default_value="/speed"),
            DeclareLaunchArgument(
                "output_progress_topic", default_value="/odom_current_progress"
            ),
            DeclareLaunchArgument(
                "clear_global_path_topic", default_value="/global_path"
            ),
            DeclareLaunchArgument("cleanup_stop_topic", default_value="/stop"),
            DeclareLaunchArgument("cleanup_stop_value", default_value="2"),
            DeclareLaunchArgument(
                "cleanup_cmd_vel_topic", default_value="/cmd_vel/final_align"
            ),
            DeclareLaunchArgument(
                "final_orientation_cmd_vel_topic",
                default_value="/cmd_vel/final_align",
            ),
            DeclareLaunchArgument(
                "cruise_speed",
                default_value="",
                description="Optional override for path_to_waypoint.cruise_speed.",
            ),
            DeclareLaunchArgument(
                "bridge_publish_rate",
                default_value="",
                description="Optional override for publish_rate. Empty uses config.",
            ),
            DeclareLaunchArgument(
                "bridge_lookahead",
                default_value="",
                description="Optional override for lookahead_distance. Empty uses config.",
            ),
            DeclareLaunchArgument(
                "bridge_min_waypoint_distance",
                default_value="",
                description="Optional override for min_waypoint_distance.",
            ),
            DeclareLaunchArgument(
                "bridge_goal_tolerance",
                default_value="",
                description="Optional override for goal_tolerance. Empty uses config.",
            ),
            DeclareLaunchArgument(
                "final_goal_tolerance",
                default_value="",
                description=(
                    "Unified final-goal tolerance override. bridge_goal_tolerance "
                    "wins when both are set."
                ),
            ),
            DeclareLaunchArgument(
                "bridge_projection_search_distance",
                default_value="",
                description="Optional override for projection_search_distance.",
            ),
            DeclareLaunchArgument(
                "bridge_loop_mode",
                default_value="",
                description="Optional override for loop_mode: none/closed/pingpong.",
            ),
            DeclareLaunchArgument(
                "bridge_max_laps",
                default_value="",
                description="Optional override for max_laps; 0 means unlimited.",
            ),
            DeclareLaunchArgument(
                "bridge_loop_pause_sec",
                default_value="",
                description="Optional override for loop_pause_sec.",
            ),
            DeclareLaunchArgument(
                "bridge_accept_repeated_path",
                default_value="",
                description="Optional override for accept_repeated_path.",
            ),
            OpaqueFunction(function=_build_path_to_waypoint),
        ]
    )
