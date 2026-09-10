"""cmd_vel postprocessor bringup.

Starts galileo_nav/cmd_vel_postprocessor with its own YAML config. Launch
arguments are optional overrides: empty values leave the YAML/node defaults
untouched, so callers can include this launch without passing parameters when
the default topic contract is enough.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BOOL_TRUE_VALUES = ("true", "1", "yes", "on")


def _as_bool(value):
    return str(value).strip().lower() in BOOL_TRUE_VALUES


def _default_config_file(pkg_galileo_nav):
    candidates = [
        os.path.join(pkg_galileo_nav, "config", "cmd_vel_postprocessor.yaml"),
        os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "config",
                "cmd_vel_postprocessor.yaml",
            )
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _load_config(config_file):
    path = os.path.expanduser(str(config_file))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _node_params(config_data, node_name):
    node_data = config_data.get(node_name, {})
    if not isinstance(node_data, dict):
        return {}
    params = node_data.get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


def _nonempty_launch_value(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _build_cmd_vel_postprocessor(context, *args, **kwargs):
    config = _load_config(_nonempty_launch_value(context, "config_file"))
    params = dict(_node_params(config, "cmd_vel_postprocessor"))

    for name in ("input_topic", "output_topic", "odom_topic"):
        value = _nonempty_launch_value(context, name)
        if value:
            params[name] = value

    for name in ("enabled", "use_sim_time"):
        value = _nonempty_launch_value(context, name)
        if value:
            params[name] = _as_bool(value)

    return [
        Node(
            package="galileo_nav",
            executable="cmd_vel_postprocessor",
            name="cmd_vel_postprocessor",
            output="screen",
            parameters=[params],
        )
    ]


def generate_launch_description():
    pkg_galileo_nav = get_package_share_directory("galileo_nav")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=_default_config_file(pkg_galileo_nav),
                description="YAML config for galileo_nav/cmd_vel_postprocessor.",
            ),
            DeclareLaunchArgument(
                "input_topic",
                default_value="",
                description="Override cmd_vel_postprocessor.input_topic when non-empty.",
            ),
            DeclareLaunchArgument(
                "output_topic",
                default_value="",
                description="Override cmd_vel_postprocessor.output_topic when non-empty.",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="",
                description="Override cmd_vel_postprocessor.odom_topic when non-empty.",
            ),
            DeclareLaunchArgument(
                "enabled",
                default_value="",
                description="Override cmd_vel_postprocessor.enabled when non-empty.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="",
                description="Override cmd_vel_postprocessor.use_sim_time when non-empty.",
            ),
            OpaqueFunction(function=_build_cmd_vel_postprocessor),
        ]
    )
