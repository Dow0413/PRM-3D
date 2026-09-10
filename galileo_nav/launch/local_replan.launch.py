"""local_replan local planner bringup for nav_main.

nav_main includes this when local:=replan. It starts a single node,
galileo_nav/local_replan, which does pure-pursuit tracking (/global_path + odom
-> /cmd_vel) AND the 20 Hz collision check that calls the ReplanPlan service
(start + goal + filtered obstacle cloud -> new path). The PRM global planner
serves the request, re-marks the obstacles and re-plans.

  ros2 launch galileo_nav nav_main.launch.py global:=prm local:=replan \
      map_name:=gazebo_sim

Channel contract:

  IN  /global_path       (nav_msgs/Path)          — PRM global path
  IN  /Odometry          (nav_msgs/Odometry)      — robot pose
  IN  /local_cloud_map   (sensor_msgs/PointCloud2) — ONE world-frame cloud for
                        the whole environment (same PCD on every floor)
  OUT /cmd_vel           (geometry_msgs/Twist)    — drive the base
  OUT replan_service     (prm_interfaces/ReplanPlan) — start+goal+障碍点云 -> 新路径

Parameters: galileo_nav/config/local_replan.yaml. ``map_name`` selects the
environment block in ``maps:``; each block lists that environment's floor PNGs
(``floors:``) with per-PNG Z bands and enable_replan switches. This file
flattens the selected block's floors into the parallel array parameters the
node consumes (ROS 2 parameters cannot carry a list-of-dicts).
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _flatten_floors(env_block):
    """maps.<env>.floors (list-of-dicts, one per floor PNG) -> parallel arrays.

    Mirrors launch_paths.flatten_floors (used for global_prm.yaml): ``index``
    orders the entries (falling back to list position), and each entry's
    z_min/z_max (the floor band, same values as global_prm.yaml), obstacle_z_min
    /obstacle_z_max (height filter relative to the robot) and enable_replan
    become floor_z_mins / floor_z_maxs / floor_obstacle_z_mins /
    floor_obstacle_z_maxs / floor_enable_replans.
    """
    floors = env_block.get("floors") if isinstance(env_block, dict) else None
    if not isinstance(floors, list) or not floors:
        return {}

    entries = []
    for position, entry in enumerate(floors):
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        entries.append((index, entry))
    entries.sort(key=lambda item: item[0])

    return {
        "floor_pngs": [str(e.get("png", "")).strip() for _, e in entries],
        "floor_z_mins": [_as_float(e.get("z_min")) for _, e in entries],
        "floor_z_maxs": [_as_float(e.get("z_max")) for _, e in entries],
        "floor_obstacle_z_mins": [_as_float(e.get("obstacle_z_min"), -0.1) for _, e in entries],
        "floor_obstacle_z_maxs": [_as_float(e.get("obstacle_z_max"), 1.2) for _, e in entries],
        "floor_enable_replans": [
            str(e.get("enable_replan", True)).lower() not in ("false", "0", "no", "off")
            for _, e in entries
        ],
    }


def _merge_params(data, map_name):
    """local_replan.ros__parameters + the selected environment block (scalars
    override; its floors are flattened into array parameters)."""
    params = {}
    node_data = data.get("local_replan", {})
    if isinstance(node_data, dict):
        ros_params = node_data.get("ros__parameters", {})
        if isinstance(ros_params, dict):
            params.update(ros_params)

    maps = data.get("maps", {})
    env_block = maps.get(map_name, {}) if isinstance(maps, dict) else {}
    if isinstance(env_block, dict):
        for key, value in env_block.items():
            if key != "floors":
                params[key] = value
        params.update(_flatten_floors(env_block))

    return params


def _override(params, key, performed_value):
    value = str(performed_value).strip()
    if value:
        params[key] = value
    return params


def _build_nodes(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context).strip()
    map_name = LaunchConfiguration("map_name").perform(context).strip()

    try:
        with open(params_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError:
        data = {}

    # map_name from the launch arg wins; otherwise fall back to the yaml default.
    if not map_name:
        node_data = data.get("local_replan", {})
        ros_params = node_data.get("ros__parameters", {}) if isinstance(node_data, dict) else {}
        if isinstance(ros_params, dict):
            map_name = str(ros_params.get("map_name", "")).strip()

    params = _merge_params(data, map_name)

    # Topic overrides threaded by nav_main (empty -> keep yaml values).
    _override(params, "path_topic", LaunchConfiguration("path_topic").perform(context))
    _override(params, "odom_topic", LaunchConfiguration("odom_topic").perform(context))
    _override(params, "cmd_topic", LaunchConfiguration("cmd_topic").perform(context))
    _override(params, "obstacle_cloud_topic", LaunchConfiguration("cloud_topic").perform(context))
    _override(
        params, "replan_service",
        LaunchConfiguration("replan_service").perform(context),
    )

    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    params["use_sim_time"] = use_sim_time.strip().lower() in ("true", "1", "yes", "on")

    node = Node(
        package="galileo_nav",
        executable="local_replan",
        name="local_replan",
        output="screen",
        parameters=[params],
    )
    return [node]


def generate_launch_description():
    galileo_nav_share = get_package_share_directory("galileo_nav")
    default_params = os.path.join(galileo_nav_share, "config", "local_replan.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to local_replan.yaml.",
    )
    map_name_arg = DeclareLaunchArgument(
        "map_name",
        default_value="",
        description="Selects the maps.<map_name> environment block (per-PNG floors).",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use /clock (rosbag playback).",
    )
    path_topic_arg = DeclareLaunchArgument(
        "path_topic", default_value="", description="Override path_topic in YAML."
    )
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic", default_value="", description="Override odom_topic in YAML."
    )
    cmd_topic_arg = DeclareLaunchArgument(
        "cmd_topic", default_value="", description="Override cmd_topic in YAML."
    )
    cloud_topic_arg = DeclareLaunchArgument(
        "cloud_topic", default_value="", description="Override obstacle_cloud_topic in YAML."
    )
    replan_service_arg = DeclareLaunchArgument(
        "replan_service", default_value="",
        description="Override replan_service in YAML.",
    )

    return LaunchDescription(
        [
            params_file_arg,
            map_name_arg,
            use_sim_time_arg,
            path_topic_arg,
            odom_topic_arg,
            cmd_topic_arg,
            cloud_topic_arg,
            replan_service_arg,
            OpaqueFunction(function=_build_nodes),
        ]
    )
