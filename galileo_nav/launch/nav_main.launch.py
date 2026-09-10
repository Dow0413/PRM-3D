"""
Unified Galileo navigation main launch — thin selector.

nav_main selects a global x local planner combo and delegates the actual
bringup to each planner's OWN launch file (one IncludeLaunchDescription per
axis, via a name -> launch-file resolver). This file carries only selector args
(global, local, mode), planner-agnostic sensing bringup, and common helper nodes
such as platform bridge/status feedback/logging. Simulation backends (mujoco,
gazebo, ...) are NOT started here; run them separately (e.g. sim_mujoco.launch.py).
Selector defaults are loaded from config/nav_main.yaml; explicit launch
arguments override that config. Real lidar bringup also reads the same config
for the selected lidar/filter profile.

Canonical planner contract (the implicit interface every planner conforms to):
  global planner OUTPUT -> /global_path      (nav_msgs/Path)
  local  planner INPUT  <- /global_path
  local  planner OUTPUT -> /cmd_vel      (geometry_msgs/Twist) -> lcm_ws

When enable_cmd_vel_postprocessor is true and local:=cmu (or wzh, whose tracking
is the same CMU stack), nav_main rewires the
CMU local stack to publish /cmd_vel_nav first, then starts
galileo_nav/cmd_vel_postprocessor:
  /cmd_vel_nav -> reachability limit -> PID speed compensation
               -> accel limits -> speed limits -> min turning radius -> /cmd_vel

The contract topics are fixed defaults owned by each planner launch; nav_main
only threads selector-safe arguments such as child config paths and `mode`
(real/sim -> which local_cmu.yaml section local_cmu reads).

  ros2 launch galileo_nav nav_main.launch.py \\
      config_file:=<path/to/nav_main.yaml> \\
      global:=<pct|topo|topo_global|trg|waypoint|octo|prm> \\
      local:=<cmu|ego|sru|scan|replan|wzh> \\
      lidar:=<airy|jt128> \\
      mode:=<real|sim>            # default real
      kinematic_model:=<original|vehicle>

Unwired names (external, galileo, nav2_mppi, nav2_rpp) log a notice
listing the available planners for that axis.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, LaunchLogDir, PythonExpression
from launch_ros.actions import Node


NAV_MAIN_DEFAULTS = {
    "use_sim_time": "false",
    "global": "pct",
    "local": "cmu",
    "lidar": "airy",
    "mode": "real",
    "kinematic_model": "",
    "global_path_topic": "/global_path",
    "localization_odom_topic": "/Odometry",
    "cmu_odom_topic": "/state_estimation",
    "waypoint_topic": "/way_point",
    "map_name": "",
    "replan_service": "/replan_plan",
    "pose_goal_topic": "/goal_pose",
    "output_cmd_vel_topic": "/cmd_vel",
    "cmd_vel_nav_topic": "/cmd_vel_nav",
    "planner_status_topic": "/planner/status",
    "start_platform_bridge": "true",
    "platform_goal_topic": "/platform/nav_goal",
    "platform_status_topic": "/platform/nav_status",
    "platform_action_name": "/platform/nav_action",
    "start_topic_state_logger": "true",
    "enable_path_to_waypoint": "true",
    "enable_cmu_start_gate": "true",
    "enable_cmd_vel_bridge": "true",
    "enable_cmd_vel_postprocessor": "false",
    "final_goal_tolerance": "0.1",
}

TRUE_VALUES = "['true', '1', 'yes', 'on']"
BOOL_TRUE_VALUES = ("true", "1", "yes", "on")


def _bool_condition(name):
    return IfCondition(
        PythonExpression(["'", LaunchConfiguration(name), "'.lower() in ", TRUE_VALUES])
    )


def _as_bool(value):
    return str(value).strip().lower() in BOOL_TRUE_VALUES


def _scoped_include(src, launch_arguments=None):
    """Include a child launch without leaking its launch arguments to siblings."""
    kwargs = {}
    if launch_arguments:
        kwargs["launch_arguments"] = launch_arguments.items()
    return GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(src),
                **kwargs,
            )
        ],
    )


def _include(src, launch_arguments=None):
    """Include a child launch in the current context."""
    kwargs = {}
    if launch_arguments:
        kwargs["launch_arguments"] = launch_arguments.items()
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(src), **kwargs)


def _load_config(config_file):
    path = os.path.expanduser(str(config_file))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _nav_main_config(config_data):
    section = config_data.get("nav_main")
    if not isinstance(section, dict):
        return config_data

    params = section.get("ros__parameters")
    return params if isinstance(params, dict) else section


def _launch_config_string(value):
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return str(value)


def _node_params(config_data, node_name):
    node_data = config_data.get(node_name, {})
    if not isinstance(node_data, dict):
        return {}
    params = node_data.get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


def _apply_nav_main_config(context, *args, **kwargs):
    config_file = LaunchConfiguration("config_file").perform(context)
    config = _nav_main_config(_load_config(config_file))

    actions = []
    for name, default_value in NAV_MAIN_DEFAULTS.items():
        launch_value = LaunchConfiguration(name).perform(context).strip()
        if launch_value:
            continue

        config_value = config.get(name, default_value)
        actions.append(
            SetLaunchConfiguration(name, _launch_config_string(config_value))
        )
    return actions


def _configure_cmd_vel_postprocessor_chain(context, *args, **kwargs):
    local_name = LaunchConfiguration("local").perform(context).strip().lower()
    postprocessor_enabled = _as_bool(
        LaunchConfiguration("enable_cmd_vel_postprocessor").perform(context)
    )
    output_topic = (
        LaunchConfiguration("output_cmd_vel_topic").perform(context).strip()
        or "/cmd_vel"
    )
    nav_topic = (
        LaunchConfiguration("cmd_vel_nav_topic").perform(context).strip()
        or "/cmd_vel_nav"
    )

    local_output_topic = output_topic
    if postprocessor_enabled and local_name in ("cmu", "wzh"):
        local_output_topic = nav_topic

    return [
        SetLaunchConfiguration("output_cmd_vel_topic", output_topic),
        SetLaunchConfiguration("cmd_vel_nav_topic", nav_topic),
        SetLaunchConfiguration("local_output_cmd_vel_topic", local_output_topic),
    ]


def _include_cmd_vel_postprocessor(context, *args, **kwargs):
    local_name = LaunchConfiguration("local").perform(context).strip().lower()
    postprocessor_enabled = _as_bool(
        LaunchConfiguration("enable_cmd_vel_postprocessor").perform(context)
    )
    if not postprocessor_enabled or local_name not in ("cmu", "wzh"):
        return []

    pkg_galileo_nav = get_package_share_directory("galileo_nav")
    child_launch_arguments = {
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "input_topic": LaunchConfiguration("cmd_vel_nav_topic"),
        "output_topic": LaunchConfiguration("output_cmd_vel_topic"),
        "odom_topic": LaunchConfiguration("localization_odom_topic"),
    }

    return [
        _scoped_include(
            os.path.join(
                pkg_galileo_nav, "launch", "cmd_vel_postprocessor.launch.py"
            ),
            launch_arguments=child_launch_arguments,
        )
    ]


def _include_platform_nav_bridge(context, *args, **kwargs):
    if not _as_bool(LaunchConfiguration("start_platform_bridge").perform(context)):
        return []

    pkg_platform_nav_bridge = get_package_share_directory("platform_nav_bridge")
    return [
        _scoped_include(
            os.path.join(
                pkg_platform_nav_bridge, "launch", "platform_nav_bridge.launch.py"
            ),
            launch_arguments={
                "platform_goal_topic": LaunchConfiguration("platform_goal_topic"),
                "platform_status_topic": LaunchConfiguration("platform_status_topic"),
                "platform_action_name": LaunchConfiguration("platform_action_name"),
                "goal_pose_topic": LaunchConfiguration("pose_goal_topic"),
                "planner_status_topic": LaunchConfiguration("planner_status_topic"),
                "global_path_topic": LaunchConfiguration("global_path_topic"),
                "odom_topic": LaunchConfiguration("cmu_odom_topic"),
                "waypoint_topic": LaunchConfiguration("waypoint_topic"),
                "cmd_vel_topic": LaunchConfiguration("output_cmd_vel_topic"),
                "planner_type": LaunchConfiguration("global"),
                "goal_reached_tolerance": LaunchConfiguration(
                    "final_goal_tolerance"
                ),
            },
        )
    ]


def _resolve_planner(context, axis, sources):
    """Include exactly one planner launch (name -> launch file) for an axis.

    Returns a LogInfo notice for any name not in ``sources`` (unwired planners).
    """
    name = LaunchConfiguration(axis).perform(context).strip().lower()
    src = sources.get(name)
    if src is None:
        available = ", ".join(sorted(sources)) or "(none)"
        return [LogInfo(msg=f"{axis}='{name}' not wired (available: {available})")]
    return [_scoped_include(src)]


def _resolve_global(context, *args, **kwargs):
    pkg_galileo_nav = get_package_share_directory("galileo_nav")
    sources = {
        "pct": os.path.join(pkg_galileo_nav, "launch", "global_pct.launch.py"),
        "trg": os.path.join(pkg_galileo_nav, "launch", "global_trg.launch.py"),
        "topo": os.path.join(pkg_galileo_nav, "launch", "global_topo.launch.py"),
        "topo_global": os.path.join(pkg_galileo_nav, "launch", "global_topo_global.launch.py"),
        "waypoint": os.path.join(pkg_galileo_nav, "launch", "global_waypoint.launch.py"),
        "octo": os.path.join(pkg_galileo_nav, "launch", "global_octo.launch.py"),
        "prm": os.path.join(pkg_galileo_nav, "launch", "global_prm.launch.py"),
    }
    child_launch_arguments = {
        "topo": {
            "config_file": os.path.join(pkg_galileo_nav, "config", "global_topo.yaml"),
        },
        "topo_global": {
            "config_file": os.path.join(
                pkg_galileo_nav, "config", "global_topo_global.yaml"
            )
        },
        "waypoint": {
            "config_file": os.path.join(
                pkg_galileo_nav, "config", "global_waypoint.yaml"
            )
        },
        "prm": {
            "config_file": os.path.join(
                pkg_galileo_nav, "config", "global_prm.yaml"
            )
        },
    }

    name = LaunchConfiguration("global").perform(context).strip().lower()
    src = sources.get(name)
    if src is None:
        available = ", ".join(sorted(sources)) or "(none)"
        return [LogInfo(msg=f"global='{name}' not wired (available: {available})")]
    return [_scoped_include(src, launch_arguments=child_launch_arguments.get(name))]


def _resolve_local(context, *args, **kwargs):
    # Inlined (not via _resolve_planner) so the local axis alone can thread
    # nav_main's `mode` into local_cmu (which selects local_cmu.yaml's
    # real/sim section). The global axis stays generic.
    sources = {
        "cmu": os.path.join(
            get_package_share_directory("galileo_nav"), "launch", "local_cmu.launch.py"
        ),
        "ego": os.path.join(
            get_package_share_directory("galileo_nav"), "launch", "local_ego.launch.py"
        ),
        "sru": os.path.join(
            get_package_share_directory("galileo_nav"), "launch", "local_sru.launch.py"
        ),
        "scan": os.path.join(
            get_package_share_directory("galileo_nav"), "launch", "local_scan.launch.py"
        ),
        "replan": os.path.join(
            get_package_share_directory("galileo_nav"), "launch", "local_replan.launch.py"
        ),
        "wzh": os.path.join(
            get_package_share_directory("galileo_nav"), "launch", "local_wzh.launch.py"
        ),
    }
    name = LaunchConfiguration("local").perform(context).strip().lower()
    src = sources.get(name)
    if src is None:
        available = ", ".join(sorted(sources)) or "(none)"
        return [LogInfo(msg=f"local='{name}' not wired (available: {available})")]

    launch_arguments = {}
    if name == "cmu":
        launch_arguments.update(
            {
                "mode": LaunchConfiguration("mode"),
                "kinematic_model": LaunchConfiguration("kinematic_model"),
                "map_name": LaunchConfiguration("map_name"),
                "global_path_topic": LaunchConfiguration("global_path_topic"),
                "localization_odom_topic": LaunchConfiguration(
                    "localization_odom_topic"
                ),
                "cmu_odom_topic": LaunchConfiguration("cmu_odom_topic"),
                "cmu_input_scan_topic": LaunchConfiguration("cmu_input_scan_topic"),
                "output_cmd_vel_topic": LaunchConfiguration(
                    "local_output_cmd_vel_topic"
                ),
                "enable_path_to_waypoint": LaunchConfiguration(
                    "enable_path_to_waypoint"
                ),
                "enable_cmu_start_gate": LaunchConfiguration("enable_cmu_start_gate"),
                "enable_cmd_vel_bridge": LaunchConfiguration("enable_cmd_vel_bridge"),
                "final_goal_tolerance": LaunchConfiguration("final_goal_tolerance"),
            }
        )
    if name == "replan":
        launch_arguments.update(
            {
                "map_name": LaunchConfiguration("map_name"),
                "path_topic": LaunchConfiguration("global_path_topic"),
                "odom_topic": LaunchConfiguration("localization_odom_topic"),
                "cmd_topic": LaunchConfiguration("local_output_cmd_vel_topic"),
                "cloud_topic": LaunchConfiguration("cmu_input_scan_topic"),
                "replan_service": LaunchConfiguration("replan_service"),
            }
        )
    if name == "wzh":
        # wzh = local_cmu 的全部参数（循迹/桥接链/后处理重接线）+ wzh_replan
        # 监控节点的 cloud/service 两个参数。cmd 不透传：wzh 不发 cmd_vel。
        launch_arguments.update(
            {
                "mode": LaunchConfiguration("mode"),
                "kinematic_model": LaunchConfiguration("kinematic_model"),
                "map_name": LaunchConfiguration("map_name"),
                "global_path_topic": LaunchConfiguration("global_path_topic"),
                "localization_odom_topic": LaunchConfiguration(
                    "localization_odom_topic"
                ),
                "cmu_odom_topic": LaunchConfiguration("cmu_odom_topic"),
                "cmu_input_scan_topic": LaunchConfiguration("cmu_input_scan_topic"),
                "output_cmd_vel_topic": LaunchConfiguration(
                    "local_output_cmd_vel_topic"
                ),
                "enable_path_to_waypoint": LaunchConfiguration(
                    "enable_path_to_waypoint"
                ),
                "enable_cmu_start_gate": LaunchConfiguration("enable_cmu_start_gate"),
                "enable_cmd_vel_bridge": LaunchConfiguration("enable_cmd_vel_bridge"),
                "final_goal_tolerance": LaunchConfiguration("final_goal_tolerance"),
                "cloud_topic": LaunchConfiguration("cmu_input_scan_topic"),
                "replan_service": LaunchConfiguration("replan_service"),
            }
        )
    # local_cmu contains delayed TimerAction/OpaqueFunction actions that resolve
    # its own LaunchConfigurations later. Keeping it unscoped preserves those
    # configs for delayed callbacks; it is started last, so it cannot pollute a
    # later sibling planner include.
    return [_include(src, launch_arguments=launch_arguments)]


def _build_topic_state_logger(context, *args, **kwargs):
    config = _load_config(LaunchConfiguration("config_file").perform(context))
    params = dict(_node_params(config, "topic_state_logger"))
    params.update(
        {
            "odom_topic": LaunchConfiguration("localization_odom_topic"),
            "cmd_vel_topic": LaunchConfiguration("output_cmd_vel_topic"),
            "waypoint_topic": LaunchConfiguration("waypoint_topic"),
        }
    )
    return [
        Node(
            package="galileo_nav",
            executable="topic_state_logger",
            name="topic_state_logger",
            output="screen",
            parameters=[params],
            condition=_bool_condition("start_topic_state_logger"),
        )
    ]


def generate_launch_description():
    pkg_galileo_nav = get_package_share_directory("galileo_nav")

    # ---- Core selection ----
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=os.path.join(pkg_galileo_nav, "config", "nav_main.yaml"),
        description="nav_main selector YAML config.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="",
        description="Override config/nav_main.yaml use_sim_time when non-empty.",
    )
    global_arg = DeclareLaunchArgument(
        "global",
        default_value="",
        description=(
            "Override config/nav_main.yaml global planner when non-empty: "
            "pct, topo, topo_global, trg, waypoint, octo, or prm."
        ),
    )
    local_arg = DeclareLaunchArgument(
        "local",
        default_value="",
        description=(
            "Override config/nav_main.yaml local planner when non-empty: "
            "cmu, ego, scan, sru, replan, or wzh (CMU tracking + per-floor "
            "local_way collision monitor)."
        ),
    )
    lidar_arg = DeclareLaunchArgument(
        "lidar",
        default_value="",
        description=(
            "Override config/nav_main.yaml lidar profile when non-empty: "
            "airy or jt128."
        ),
    )
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="",
        description=(
            "Override config/nav_main.yaml mode when non-empty: 'real' or 'sim'. "
            "real -> local_cmu reads local_cmu.yaml real section; "
            "sim -> the sim section. "
            "Sim backends (mujoco, gazebo, ...) are launched separately "
            "(e.g. sim_mujoco.launch.py)."
        ),
    )
    kinematic_model_arg = DeclareLaunchArgument(
        "kinematic_model",
        default_value="",
        description=(
            "Override local_cmu kinematic model profile: original/omni/legacy "
            "or vehicle/car/ackermann. Empty uses config."
        ),
    )
    global_path_topic_arg = DeclareLaunchArgument(
        "global_path_topic",
        default_value="",
        description="Canonical global planner output topic.",
    )
    localization_odom_topic_arg = DeclareLaunchArgument(
        "localization_odom_topic",
        default_value="",
        description="Localization odometry topic used by common helper nodes.",
    )
    cmu_odom_topic_arg = DeclareLaunchArgument(
        "cmu_odom_topic",
        default_value="",
        description="CMU-adapted odometry topic used by platform bridge.",
    )
    cmu_input_scan_topic_arg = DeclareLaunchArgument(
        "cmu_input_scan_topic",
        default_value="/local_cloud_map",
        description=(
            "Point cloud topic fed to the CMU local planner via loam_interface "
            "(forwarded to local_cmu.launch.py)."
        ),
    )
    map_name_arg = DeclareLaunchArgument(
        "map_name",
        default_value="",
        description=(
            "Current map name; selects local_replan.yaml maps.<map_name> "
            "(per-map Z band + enable_replan)."
        ),
    )
    replan_service_arg = DeclareLaunchArgument(
        "replan_service",
        default_value="",
        description="ReplanPlan service the local_replan planner calls for a replan.",
    )
    waypoint_topic_arg = DeclareLaunchArgument(
        "waypoint_topic",
        default_value="",
        description="Current local waypoint topic.",
    )
    pose_goal_topic_arg = DeclareLaunchArgument(
        "pose_goal_topic",
        default_value="",
        description="PoseStamped goal topic consumed by compatible global planners.",
    )
    output_cmd_vel_topic_arg = DeclareLaunchArgument(
        "output_cmd_vel_topic",
        default_value="",
        description="Final Twist command topic.",
    )
    cmd_vel_nav_topic_arg = DeclareLaunchArgument(
        "cmd_vel_nav_topic",
        default_value="",
        description=(
            "Intermediate Twist command topic used as cmd_vel_postprocessor "
            "input when local:=cmu and enable_cmd_vel_postprocessor is true."
        ),
    )
    planner_status_topic_arg = DeclareLaunchArgument(
        "planner_status_topic",
        default_value="",
        description="Unified JSON planner status topic.",
    )
    start_platform_bridge_arg = DeclareLaunchArgument(
        "start_platform_bridge",
        default_value="",
        description="Start platform_nav_bridge and global_planner_feedback.",
    )
    platform_goal_topic_arg = DeclareLaunchArgument(
        "platform_goal_topic",
        default_value="",
        description="Platform JSON goal topic.",
    )
    platform_status_topic_arg = DeclareLaunchArgument(
        "platform_status_topic",
        default_value="",
        description="Platform JSON status topic.",
    )
    platform_action_name_arg = DeclareLaunchArgument(
        "platform_action_name",
        default_value="",
        description="Platform NavigateToPose action name.",
    )
    start_topic_state_logger_arg = DeclareLaunchArgument(
        "start_topic_state_logger",
        default_value="",
        description="Start galileo_nav/topic_state_logger.",
    )
    enable_path_to_waypoint_arg = DeclareLaunchArgument(
        "enable_path_to_waypoint",
        default_value="",
        description="Start global path to CMU waypoint bridge when local:=cmu.",
    )
    enable_cmu_start_gate_arg = DeclareLaunchArgument(
        "enable_cmu_start_gate",
        default_value="",
        description="Start CMU /stop gate when local:=cmu.",
    )
    enable_cmd_vel_bridge_arg = DeclareLaunchArgument(
        "enable_cmd_vel_bridge",
        default_value="",
        description="Start CMU TwistStamped->Twist bridge and cmd_vel_mux.",
    )
    enable_cmd_vel_postprocessor_arg = DeclareLaunchArgument(
        "enable_cmd_vel_postprocessor",
        default_value="",
        description=(
            "When true with local:=cmu, route CMU cmd_vel through "
            "galileo_nav/cmd_vel_postprocessor before output_cmd_vel_topic."
        ),
    )
    final_goal_tolerance_arg = DeclareLaunchArgument(
        "final_goal_tolerance",
        default_value="",
        description=(
            "Unified final-goal tolerance in meters. Applied to "
            "platform_nav_bridge.goal_reached_tolerance and, for local:=cmu, "
            "path_to_waypoint.goal_tolerance, "
            "cmu_start_gate.goal_reached_tolerance, and pathFollower.stopDisThre."
        ),
    )

    set_ros_log_dir_to_launch_log_dir = SetEnvironmentVariable(
        name="ROS_LOG_DIR", value=LaunchLogDir()
    )

    return LaunchDescription(
        [
            # args
            config_file_arg,
            use_sim_time_arg,
            global_arg,
            local_arg,
            lidar_arg,
            mode_arg,
            kinematic_model_arg,
            global_path_topic_arg,
            localization_odom_topic_arg,
            cmu_odom_topic_arg,
            cmu_input_scan_topic_arg,
            map_name_arg,
            replan_service_arg,
            waypoint_topic_arg,
            pose_goal_topic_arg,
            output_cmd_vel_topic_arg,
            cmd_vel_nav_topic_arg,
            planner_status_topic_arg,
            start_platform_bridge_arg,
            platform_goal_topic_arg,
            platform_status_topic_arg,
            platform_action_name_arg,
            start_topic_state_logger_arg,
            enable_path_to_waypoint_arg,
            enable_cmu_start_gate_arg,
            enable_cmd_vel_bridge_arg,
            enable_cmd_vel_postprocessor_arg,
            final_goal_tolerance_arg,
            OpaqueFunction(function=_apply_nav_main_config),
            OpaqueFunction(function=_configure_cmd_vel_postprocessor_chain),

            # env
            set_ros_log_dir_to_launch_log_dir,
            OpaqueFunction(function=_include_platform_nav_bridge),
            OpaqueFunction(function=_build_topic_state_logger),
            OpaqueFunction(function=_include_cmd_vel_postprocessor),

            # planner bringup (delegated to each planner's own launch file)
            OpaqueFunction(function=_resolve_global),
            TimerAction(
                period=2.0, actions=[OpaqueFunction(function=_resolve_local)]
            ),
        ]
    )
