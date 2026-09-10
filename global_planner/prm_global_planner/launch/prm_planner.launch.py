"""PRM-3D global planner standalone bringup.

Starts prm_planner_node and loads its parameters from a YAML config. The node
publishes the canonical /global_path topic (nav_msgs/Path) with RViz markers.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from galileo_nav.launch_paths import flatten_floors, resolve_path, resolve_path_params
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_config(config_file):
    path = resolve_path(str(config_file))
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


def build(context, *args, **kwargs):
    cfg = _load_config(LaunchConfiguration("config_file").perform(context))
    params = resolve_path_params(
        flatten_floors(_node_params(cfg, "prm_planner_node")), ["connections_json"]
    )

    return [
        Node(
            package="prm_global_planner",
            executable="prm_planner_node",
            name="prm_planner_node",
            output="screen",
            parameters=[params],
        )
    ]


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory("prm_global_planner"),
        "config",
        "prm_planner_params.yaml",
    )

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_params_file,
        description="YAML parameter file for prm_planner_node.",
    )

    return LaunchDescription([config_file_arg, OpaqueFunction(function=build)])
