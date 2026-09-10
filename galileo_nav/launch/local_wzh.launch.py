"""local_wzh local planner bringup for nav_main.

wzh = CMU 循迹 + 按楼层 PNG 标签（local_way）挂载 local_replan 碰撞监控：

  1. 完整复用 local_cmu.launch.py 的 CMU 栈（terrainAnalysis/localPlanner/
     pathFollower/桥接链），循迹 100% 由 CMU 负责。唯一改动：默认
     enable_cmu_replan:=false，关掉 CMU 卡死重规划监视（wzh 的重规划由碰撞
     触发接管；需要双保险时可在本 launch 重新打开）。
  2. 额外启动 galileo_nav/wzh_replan 节点：local_replan 的监控半边（障碍
     点云过滤 + 前视碰撞检测 + ReplanPlan + 障碍入图），不发 cmd_vel。

  ros2 launch galileo_nav nav_main.launch.py global:=prm local:=wzh \\
      map_name:=gazebo_sim

楼层标签（local_wzh.yaml maps.<map_name>.floors 每层一个）：

      local_way: replan -> 该层碰撞监控开（触发重规划 + 入图，同 local_replan）
      local_way: cmu    -> 该层纯 CMU 循迹（不触发、不入图；楼梯 PNG 用这个）

本文件把选中环境的 floors 展开成并行数组参数（含 floor_local_ways）传给
wzh_replan，展开逻辑与 local_replan.launch.py 的 _flatten_floors 一致（launch
文件间不做跨文件 import，遵循 local_cmu.launch.py 复制 _flatten_replan_floors
的既有惯例）。
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _flatten_floors(env_block):
    """maps.<env>.floors -> 并行数组参数；比 local_replan.launch.py 多展一个
    local_way（cmu|replan，默认 replan）成 floor_local_ways。"""
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
        "floor_local_ways": [
            str(e.get("local_way", "replan")).strip().lower() or "replan"
            for _, e in entries
        ],
    }


def _merge_params(data, map_name):
    """wzh_replan.ros__parameters + 选中环境块（标量覆盖 + floors 展开）。"""
    params = {}
    node_data = data.get("wzh_replan", {})
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
    pkg_galileo_nav = get_package_share_directory("galileo_nav")

    # ---- 1. CMU 栈（循迹 + 桥接链，参数/节拍完全同 local:=cmu）----
    local_cmu = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_galileo_nav, "launch", "local_cmu.launch.py")
        ),
        launch_arguments={
            "mode": LaunchConfiguration("mode"),
            "kinematic_model": LaunchConfiguration("kinematic_model"),
            "map_name": LaunchConfiguration("map_name"),
            "global_path_topic": LaunchConfiguration("global_path_topic"),
            "localization_odom_topic": LaunchConfiguration("localization_odom_topic"),
            "cmu_odom_topic": LaunchConfiguration("cmu_odom_topic"),
            "cmu_input_scan_topic": LaunchConfiguration("cmu_input_scan_topic"),
            "output_cmd_vel_topic": LaunchConfiguration("output_cmd_vel_topic"),
            "enable_path_to_waypoint": LaunchConfiguration("enable_path_to_waypoint"),
            "enable_cmu_start_gate": LaunchConfiguration("enable_cmu_start_gate"),
            "enable_cmd_vel_bridge": LaunchConfiguration("enable_cmd_vel_bridge"),
            "final_goal_tolerance": LaunchConfiguration("final_goal_tolerance"),
            # wzh 的重规划走碰撞触发（wzh_replan），默认关掉 CMU 卡死监视。
            "enable_cmu_replan": LaunchConfiguration("enable_cmu_replan"),
            "replan_cloud_topic": LaunchConfiguration("replan_cloud_topic"),
        }.items(),
    )

    # ---- 2. wzh_replan 监控节点（config/local_wzh.yaml 独立调参面）----
    params_file = LaunchConfiguration("params_file").perform(context).strip()
    map_name = LaunchConfiguration("map_name").perform(context).strip()

    try:
        with open(params_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError:
        data = {}

    # map_name launch 参数优先；否则回退 yaml 里的 wzh_replan.map_name。
    if not map_name:
        node_data = data.get("wzh_replan", {})
        ros_params = node_data.get("ros__parameters", {}) if isinstance(node_data, dict) else {}
        if isinstance(ros_params, dict):
            map_name = str(ros_params.get("map_name", "")).strip()

    params = _merge_params(data, map_name)

    # 话题覆盖（nav_main thread 进来的非空值生效）。
    _override(params, "path_topic", LaunchConfiguration("global_path_topic").perform(context))
    _override(params, "odom_topic", LaunchConfiguration("localization_odom_topic").perform(context))
    _override(params, "obstacle_cloud_topic", LaunchConfiguration("cloud_topic").perform(context))
    _override(
        params, "replan_service",
        LaunchConfiguration("replan_service").perform(context),
    )
    # wzh 不发 cmd_vel；指到死话题，双保险确保永不与 CMU 栈抢 /cmd_vel。
    params["cmd_topic"] = "/wzh/cmd_vel_unused"

    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    params["use_sim_time"] = use_sim_time.strip().lower() in ("true", "1", "yes", "on")

    wzh_replan = Node(
        package="galileo_nav",
        executable="wzh_replan",
        name="wzh_replan",
        output="screen",
        parameters=[params],
    )

    return [local_cmu, wzh_replan]


def generate_launch_description():
    galileo_nav_share = get_package_share_directory("galileo_nav")
    default_params = os.path.join(galileo_nav_share, "config", "local_wzh.yaml")

    return LaunchDescription(
        [
            # ---- local_cmu 透传参数（默认值同 local_cmu.launch.py）----
            DeclareLaunchArgument(
                "mode", default_value="real",
                description="local_cmu.yaml real/sim 参数段选择（nav_main 传 sim）。",
            ),
            DeclareLaunchArgument(
                "kinematic_model", default_value="",
                description="local_cmu 运动学模型 profile，空用配置默认。",
            ),
            DeclareLaunchArgument("global_path_topic", default_value="/global_path"),
            DeclareLaunchArgument("localization_odom_topic", default_value="/Odometry"),
            DeclareLaunchArgument("cmu_odom_topic", default_value="/state_estimation"),
            DeclareLaunchArgument("cmu_input_scan_topic", default_value="/local_cloud_map"),
            DeclareLaunchArgument("output_cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("enable_path_to_waypoint", default_value="true"),
            DeclareLaunchArgument("enable_cmu_start_gate", default_value="true"),
            DeclareLaunchArgument("enable_cmd_vel_bridge", default_value="true"),
            DeclareLaunchArgument("final_goal_tolerance", default_value=""),
            DeclareLaunchArgument(
                "map_name", default_value="",
                description="选 local_replan.yaml maps.<map_name> 楼层块。",
            ),
            # ---- wzh 专属 ----
            DeclareLaunchArgument(
                "enable_cmu_replan", default_value="false",
                description=(
                    "CMU 卡死重规划监视（cmu_replan）。wzh 默认关（重规划由 "
                    "wzh_replan 碰撞触发接管）；true = 双监视并存，都调同一个服务。"
                ),
            ),
            DeclareLaunchArgument("replan_cloud_topic", default_value=""),
            DeclareLaunchArgument(
                "params_file", default_value=default_params,
                description="Path to local_wzh.yaml（wzh 独立配置）。",
            ),
            DeclareLaunchArgument("cloud_topic", default_value=""),
            DeclareLaunchArgument("replan_service", default_value=""),
            DeclareLaunchArgument(
                "use_sim_time", default_value="false",
                description="Use /clock (rosbag playback).",
            ),
            OpaqueFunction(function=_build_nodes),
        ]
    )
