"""CMU local planner bringup — conforms to the unified planner contract.

Starts the CMU localPlanner/pathFollower nodes and wraps them with the
galileo_nav glue nodes (path_to_waypoint.launch.py, twist_stamped_to_twist,
cmd_vel_mux) and cmu_start_gate so it consumes the global path and emits
/cmd_vel (Twist), without modifying CMU C++.
Also brings up the CMU-specific perception pipeline (sensor_scan_generation +
terrain_analysis + terrain_analysis_ext) and the loam_interface odom bridge,
since only the CMU local planner consumes them. All tunable parameters are read
from config/local_cmu.yaml and config/path_to_waypoint.yaml.
Standalone-runnable; nav_main includes this when local:=cmu. Contract topics are
fixed defaults, not threaded.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    LaunchLogDir,
    PythonExpression,
)
from launch_ros.actions import Node


TRUE_VALUES = "['true', '1', 'yes', 'on']"
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
    "localPlanner": ("maxSpeed", "autonomySpeed"),
    "pathFollower": ("maxSpeed", "autonomySpeed"),
}
MODEL_SPEED_NODE_PARAMS = {
    "pathFollower": (
        ("max_yaw_rate", "maxYawRate"),
        ("max_accel", "maxAccel"),
        ("max_lateral_speed", "maxLateralSpeed"),
        ("max_lateral_accel", "maxLateralAccel"),
    ),
}


def _bool_condition(name):
    return IfCondition(
        PythonExpression(["'", LaunchConfiguration(name), "'.lower() in ", TRUE_VALUES])
    )


def _load_config_params(config_file):
    with open(os.path.expanduser(config_file), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _node_params_from_mapping(mapping, node_name):
    node_data = mapping.get(node_name, {})
    if not isinstance(node_data, dict):
        return {}
    params = node_data.get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


def _as_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _flatten_replan_floors(env_block):
    """local_replan.yaml 的 maps.<env>.floors（每层一个 dict）-> 并行数组参数。

    与 local_replan.launch.py 的 _flatten_floors 完全一致：cmu_replan 与
    local_replan 共用同一份楼层配置（local_replan.yaml 的 maps: 段是唯一
    数据源），换环境只改一处。
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


def _replan_floor_params(map_name):
    """local_replan.yaml maps.<map_name> 段 -> cmu_replan 的分楼层参数。

    map_name 为空时回退到 local_replan.yaml 里 local_replan.map_name 的默认值。
    读不到（无该环境/无 maps 段）返回 {}：cmu_replan 会退回全局
    obstacle_z_min/max 兜底 Z 带。"""
    share = get_package_share_directory("galileo_nav")
    path = os.path.join(share, "config", "local_replan.yaml")
    try:
        data = _load_config_params(path)
    except OSError:
        return {}
    if not map_name:
        node_data = data.get("local_replan", {})
        ros_params = node_data.get("ros__parameters", {}) if isinstance(node_data, dict) else {}
        if isinstance(ros_params, dict):
            map_name = str(ros_params.get("map_name", "")).strip()
    env_block = data.get("maps", {}).get(map_name, {}) if map_name else {}
    if not isinstance(env_block, dict):
        return {}
    return _flatten_replan_floors(env_block)


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


def _kinematic_linear_speed(config_data, kinematic_model):
    model_name = _normalize_kinematic_model(kinematic_model)
    if not model_name:
        return None
    model_data = config_data.get("kinematic_models", {}).get(model_name, {})
    if not isinstance(model_data, dict):
        return None
    return _model_value(model_data, "linear_speed")


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


def _section_start_gate_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data, section, "cmu_start_gate", kinematic_model=kinematic_model
    )


def _section_cmd_bridge_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data, section, "cmu_cmd_to_cmd_vel", kinematic_model=kinematic_model
    )


def _section_cmd_vel_mux_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data, section, "cmd_vel_mux", kinematic_model=kinematic_model
    )


def _section_terrain_analysis_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data, section, "terrainAnalysis", kinematic_model=kinematic_model
    )


def _section_terrain_analysis_ext_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data, section, "terrainAnalysisExt", kinematic_model=kinematic_model
    )


def _section_local_planner_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data, section, "localPlanner", kinematic_model=kinematic_model
    )


def _section_path_follower_params(config_data, section, kinematic_model=""):
    return _section_node_params(
        config_data, section, "pathFollower", kinematic_model=kinematic_model
    )


def _default_local_cmu_config_file(pkg_galileo_nav):
    candidates = [
        os.path.join(pkg_galileo_nav, "config", "local_cmu.yaml"),
        os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "config",
                "local_cmu.yaml",
            )
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _default_path_to_waypoint_config_file(pkg_galileo_nav):
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


def _resolve_path_folder(raw_path_folder, pkg_local_planner, default_subdir):
    value = "" if raw_path_folder is None else str(raw_path_folder).strip()
    if not value:
        return os.path.join(pkg_local_planner, "paths", default_subdir)
    if os.path.isabs(value):
        return value
    if value == "paths":
        return os.path.join(pkg_local_planner, "paths")
    return os.path.join(pkg_local_planner, "paths", value)


def _default_path_folder(
    config_file, section, pkg_local_planner, default_subdir="0_35", kinematic_model=""
):
    try:
        config_data = _load_config_params(config_file)
    except (OSError, yaml.YAMLError):
        return os.path.join(pkg_local_planner, "paths", default_subdir)

    params = _section_local_planner_params(config_data, section, kinematic_model)
    return _resolve_path_folder(
        params.get("pathFolder", default_subdir), pkg_local_planner, default_subdir
    )


def generate_launch_description():
    pkg_galileo_nav = get_package_share_directory("galileo_nav")
    pkg_local_planner = get_package_share_directory("local_planner")
    pkg_sensor_scan_generation = get_package_share_directory("sensor_scan_generation")
    default_local_cmu_config_file = _default_local_cmu_config_file(pkg_galileo_nav)
    default_path_to_waypoint_config_file = _default_path_to_waypoint_config_file(
        pkg_galileo_nav
    )
    path_to_waypoint_launch_file = os.path.join(
        pkg_galileo_nav, "launch", "path_to_waypoint.launch.py"
    )

    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="real",
        description=(
            "Param section selector for config/local_cmu.yaml: 'real' or 'sim'. "
            "Unknown values fall back to 'real'. nav_main passes 'sim' when "
            "chassis:=mujoco."
        ),
    )
    kinematic_model_arg = DeclareLaunchArgument(
        "kinematic_model",
        default_value="",
        description=(
            "Kinematic model profile from local_cmu/path_to_waypoint kinematic_models: "
            "original/omni/legacy or vehicle/car/ackermann. Empty uses config."
        ),
    )
    global_path_topic_arg = DeclareLaunchArgument(
        "global_path_topic", default_value="/global_path"
    )

    # Galileo localization outputs. localization_airy_nav.launch.py publishes these.
    localization_odom_topic_arg = DeclareLaunchArgument(
        "localization_odom_topic", default_value="/Odometry"
    )
    cmu_input_scan_topic_arg = DeclareLaunchArgument(
        "cmu_input_scan_topic", default_value="/local_cloud_map"
    )

    # CMU-adapted odometry topic created by loam_interface. CMU point cloud
    # output is fixed by loam_interface/local_planner as /registered_scan.
    cmu_odom_topic_arg = DeclareLaunchArgument(
        "cmu_odom_topic", default_value="/state_estimation"
    )

    enable_loam_interface_arg = DeclareLaunchArgument(
        "enable_loam_interface", default_value="true"
    )

    # CMU terrain/local planner.
    enable_terrain_analysis_arg = DeclareLaunchArgument(
        "enable_terrain_analysis", default_value="true"
    )
    enable_terrain_analysis_ext_arg = DeclareLaunchArgument(
        "enable_terrain_analysis_ext", default_value="true"
    )
    enable_sensor_scan_generation_arg = DeclareLaunchArgument(
        "enable_sensor_scan_generation", default_value="true"
    )
    enable_cmu_local_planner_arg = DeclareLaunchArgument(
        "enable_cmu_local_planner", default_value="true"
    )
    # Path-to-waypoint bridge and stop gate.
    local_cmu_config_file_arg = DeclareLaunchArgument(
        "local_cmu_config_file",
        default_value=default_local_cmu_config_file,
        description="Unified YAML config for the CMU local stack.",
    )
    path_to_waypoint_config_file_arg = DeclareLaunchArgument(
        "path_to_waypoint_config_file",
        default_value=default_path_to_waypoint_config_file,
        description=(
            "Optional YAML config for the global Path -> waypoint bridge. "
            "Empty uses path_to_waypoint.launch.py defaults."
        ),
    )
    enable_path_to_waypoint_arg = DeclareLaunchArgument(
        "enable_path_to_waypoint",
        default_value="true",
        description="Start the global path to CMU waypoint bridge.",
    )
    final_goal_tolerance_arg = DeclareLaunchArgument(
        "final_goal_tolerance",
        default_value="",
        description=(
            "Unified final-goal tolerance override. Empty uses local_cmu.yaml "
            "and path_to_waypoint.yaml values."
        ),
    )
    enable_cmu_start_gate_arg = DeclareLaunchArgument(
        "enable_cmu_start_gate", default_value="true"
    )

    # CMU stuck-detection replan monitor (galileo_nav/cmu_replan). Params live
    # in local_cmu.yaml's replan: section; enable_replan=false there keeps the
    # node inert so CMU behaves exactly as before.
    enable_cmu_replan_arg = DeclareLaunchArgument(
        "enable_cmu_replan", default_value="true"
    )
    replan_cloud_topic_arg = DeclareLaunchArgument(
        "replan_cloud_topic",
        default_value="",
        description="Override replan:obstacle_cloud_topic in local_cmu.yaml.",
    )
    map_name_arg = DeclareLaunchArgument(
        "map_name",
        default_value="",
        description=(
            "Selects local_replan.yaml maps.<map_name> floors for cmu_replan "
            "(per-floor Z bands / enable_replan switches)."
        ),
    )

    # Command output.
    enable_cmd_vel_bridge_arg = DeclareLaunchArgument(
        "enable_cmd_vel_bridge", default_value="true"
    )
    cmu_cmd_vel_stamped_topic_arg = DeclareLaunchArgument(
        "cmu_cmd_vel_stamped_topic", default_value="/cmu/cmd_vel_stamped"
    )
    cmu_path_topic_arg = DeclareLaunchArgument(
        "cmu_path_topic", default_value="/cmu/path"
    )
    output_cmd_vel_topic_arg = DeclareLaunchArgument(
        "output_cmd_vel_topic", default_value="/cmd_vel"
    )

    global_path_topic = LaunchConfiguration("global_path_topic")
    cmu_odom_topic = LaunchConfiguration("cmu_odom_topic")

    set_ros_log_dir_to_launch_log_dir = SetEnvironmentVariable(
        name="ROS_LOG_DIR",
        value=LaunchLogDir(),
    )

    loam_interface = Node(
        package="loam_interface",
        executable="loamInterface",
        name="loamInterface",
        output="screen",
        parameters=[
            {
                "stateEstimationTopic": LaunchConfiguration("localization_odom_topic"),
                "registeredScanTopic": LaunchConfiguration(
                    "cmu_input_scan_topic"
                ),
                "flipStateEstimation": False,
                "flipRegisteredScan": False,
                "sendTF": True,
                "reverseTF": False,
            }
        ],
        condition=_bool_condition("enable_loam_interface"),
    )

    def build_terrain_analysis(context, *args, **kwargs):
        config_file = LaunchConfiguration("local_cmu_config_file").perform(context)
        config_data = _load_config_params(config_file)
        mode = LaunchConfiguration("mode").perform(context).strip().lower()
        section = "sim" if mode == "sim" else "real"
        kinematic_model = _selected_kinematic_model(context, config_data, section)
        params = _section_terrain_analysis_params(
            config_data, section, kinematic_model
        )

        return [
            Node(
                package="terrain_analysis",
                executable="terrainAnalysis",
                name="terrainAnalysis",
                output="screen",
                parameters=[params],
                condition=_bool_condition("enable_terrain_analysis"),
            )
        ]

    terrain_analysis = OpaqueFunction(function=build_terrain_analysis)

    def build_terrain_analysis_ext(context, *args, **kwargs):
        config_file = LaunchConfiguration("local_cmu_config_file").perform(context)
        config_data = _load_config_params(config_file)
        mode = LaunchConfiguration("mode").perform(context).strip().lower()
        section = "sim" if mode == "sim" else "real"
        kinematic_model = _selected_kinematic_model(context, config_data, section)
        params = _section_terrain_analysis_ext_params(
            config_data, section, kinematic_model
        )

        return [
            Node(
                package="terrain_analysis_ext",
                executable="terrainAnalysisExt",
                name="terrainAnalysisExt",
                output="screen",
                parameters=[params],
                condition=_bool_condition("enable_terrain_analysis_ext"),
            )
        ]

    terrain_analysis_ext = OpaqueFunction(function=build_terrain_analysis_ext)

    sensor_scan_generation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_sensor_scan_generation,
                "launch",
                "sensor_scan_generation.launch.py",
            )
        ),
        condition=_bool_condition("enable_sensor_scan_generation"),
    )

    def build_cmu_local_planner(context, *args, **kwargs):
        config_file = LaunchConfiguration("local_cmu_config_file").perform(context)
        config_data = _load_config_params(config_file)
        mode = LaunchConfiguration("mode").perform(context).strip().lower()
        section = "sim" if mode == "sim" else "real"
        kinematic_model = _selected_kinematic_model(context, config_data, section)
        local_planner_params = _section_local_planner_params(
            config_data, section, kinematic_model
        )
        local_planner_node_params = dict(local_planner_params)
        local_planner_node_params["pathFolder"] = _default_path_folder(
            config_file, section, pkg_local_planner, kinematic_model=kinematic_model
        )

        path_follower_node_params = _section_path_follower_params(
            config_data, section, kinematic_model
        )
        final_goal_tolerance = (
            LaunchConfiguration("final_goal_tolerance").perform(context).strip()
        )
        if final_goal_tolerance:
            path_follower_node_params["stopDisThre"] = float(final_goal_tolerance)

        sensor_offset_x = str(
            float(local_planner_node_params.get("sensorOffsetX", 0.0))
        )
        sensor_offset_y = str(
            float(local_planner_node_params.get("sensorOffsetY", 0.0))
        )

        return [
            Node(
                package="local_planner",
                executable="localPlanner",
                name="localPlanner",
                output="screen",
                parameters=[local_planner_node_params],
                remappings=[
                    ("/path", LaunchConfiguration("cmu_path_topic")),
                ],
                condition=_bool_condition("enable_cmu_local_planner"),
            ),
            Node(
                package="local_planner",
                executable="pathFollower",
                name="pathFollower",
                output="screen",
                parameters=[path_follower_node_params],
                remappings=[
                    ("/path", LaunchConfiguration("cmu_path_topic")),
                    ("/cmd_vel", LaunchConfiguration("cmu_cmd_vel_stamped_topic")),
                ],
                condition=_bool_condition("enable_cmu_local_planner"),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="vehicleTransPublisher",
                arguments=[
                    str(-float(sensor_offset_x)),
                    str(-float(sensor_offset_y)),
                    "0",
                    "0",
                    "0",
                    "0",
                    "/sensor",
                    "/vehicle",
                ],
                condition=_bool_condition("enable_cmu_local_planner"),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="sensorTransPublisher",
                arguments=[
                    "0",
                    "0",
                    "0",
                    "-1.5707963",
                    "0",
                    "-1.5707963",
                    "/sensor",
                    "/camera",
                ],
                condition=_bool_condition("enable_cmu_local_planner"),
            ),
        ]

    cmu_local_planner = OpaqueFunction(function=build_cmu_local_planner)

    def build_cmu_start_gate(context, *args, **kwargs):
        config_file = LaunchConfiguration("local_cmu_config_file").perform(context)
        config_data = _load_config_params(config_file)
        mode = LaunchConfiguration("mode").perform(context).strip().lower()
        section = "sim" if mode == "sim" else "real"
        cmu_kinematic_model = _selected_kinematic_model(context, config_data, section)

        gate_params = _section_start_gate_params(
            config_data, section, cmu_kinematic_model
        )
        final_goal_tolerance = (
            LaunchConfiguration("final_goal_tolerance").perform(context).strip()
        )
        if final_goal_tolerance:
            gate_params["goal_reached_tolerance"] = float(final_goal_tolerance)

        cmu_start_gate = Node(
            package="galileo_nav",
            executable="cmu_start_gate",
            name="cmu_start_gate",
            output="screen",
            parameters=[
                gate_params,
                {
                    "waypoint_topic": "/way_point",
                    "global_path_topic": global_path_topic,
                    "odom_topic": cmu_odom_topic,
                    "stop_topic": "/stop",
                },
            ],
            condition=_bool_condition("enable_cmu_start_gate"),
        )

        return [cmu_start_gate]

    cmu_start_gate = OpaqueFunction(function=build_cmu_start_gate)

    def build_replan_monitor(context, *args, **kwargs):
        """CMU 卡死重规划监视节点（galileo_nav/cmu_replan，节点名 replan）。

        触发逻辑与 local_replan 不同：不看路径是否被点云挡住，而是机器狗在
        use_radius 内持续 use_time 不动 → 判定 CMU 无法求解 → 调同一个
        ReplanPlan 服务。其余（分楼层 Z 带/goal 降级通道/入图）与 local_replan
        一致：标量参数在 local_cmu.yaml 的 replan: 段，分楼层配置展开自
        local_replan.yaml 的 maps.<map_name> 段（唯一数据源）。
        """
        config_file = LaunchConfiguration("local_cmu_config_file").perform(context)
        config_data = _load_config_params(config_file)
        params = _node_params_from_mapping(config_data, "replan")
        params.update(
            {
                "path_topic": global_path_topic,
                "odom_topic": LaunchConfiguration("localization_odom_topic"),
            }
        )
        params.update(
            _replan_floor_params(
                LaunchConfiguration("map_name").perform(context).strip()
            )
        )
        replan_cloud_topic = (
            LaunchConfiguration("replan_cloud_topic").perform(context).strip()
        )
        if replan_cloud_topic:
            params["obstacle_cloud_topic"] = replan_cloud_topic

        return [
            Node(
                package="galileo_nav",
                executable="cmu_replan",
                name="replan",
                output="screen",
                parameters=[params],
                condition=_bool_condition("enable_cmu_replan"),
            )
        ]

    replan_monitor = OpaqueFunction(function=build_replan_monitor)

    def build_path_tracking_nodes(context, *args, **kwargs):
        config_file = LaunchConfiguration("local_cmu_config_file").perform(context)
        config_data = _load_config_params(config_file)
        mode = LaunchConfiguration("mode").perform(context).strip().lower()
        section = "sim" if mode == "sim" else "real"
        kinematic_model = _selected_kinematic_model(context, config_data, section)
        linear_speed = _kinematic_linear_speed(config_data, kinematic_model)
        path_to_waypoint_config_file = (
            LaunchConfiguration("path_to_waypoint_config_file")
            .perform(context)
            .strip()
            or default_path_to_waypoint_config_file
        )
        path_to_waypoint_args = {
            "config_file": path_to_waypoint_config_file,
            "mode": LaunchConfiguration("mode"),
            "kinematic_model": LaunchConfiguration("kinematic_model"),
            "enable_path_to_waypoint": LaunchConfiguration(
                "enable_path_to_waypoint"
            ),
            "final_goal_tolerance": LaunchConfiguration("final_goal_tolerance"),
        }
        if linear_speed is not None:
            path_to_waypoint_args["cruise_speed"] = str(linear_speed)

        path_to_waypoint = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(path_to_waypoint_launch_file),
            launch_arguments=path_to_waypoint_args.items(),
        )

        return [TimerAction(period=2.5, actions=[path_to_waypoint])]

    path_tracking_nodes = OpaqueFunction(function=build_path_tracking_nodes)

    def build_cmd_vel_bridge(context, *args, **kwargs):
        config_file = LaunchConfiguration("local_cmu_config_file").perform(context)
        config_data = _load_config_params(config_file)
        mode = LaunchConfiguration("mode").perform(context).strip().lower()
        section = "sim" if mode == "sim" else "real"
        kinematic_model = _selected_kinematic_model(context, config_data, section)
        params = _section_cmd_bridge_params(config_data, section, kinematic_model)
        mux_params = _section_cmd_vel_mux_params(config_data, section, kinematic_model)
        cmu_cmd_vel_topic = str(
            mux_params.get("fallback_topic", "/cmd_vel/cmu")
        ).strip() or "/cmd_vel/cmu"
        final_align_cmd_vel_topic = str(
            mux_params.get("override_topic", "/cmd_vel/final_align")
        ).strip() or "/cmd_vel/final_align"
        params.update(
            {
                "input_topic": LaunchConfiguration("cmu_cmd_vel_stamped_topic"),
                "output_topic": cmu_cmd_vel_topic,
            }
        )
        mux_params.update(
            {
                "fallback_topic": cmu_cmd_vel_topic,
                "override_topic": final_align_cmd_vel_topic,
                "output_topic": LaunchConfiguration("output_cmd_vel_topic"),
            }
        )
        return [
            Node(
                package="galileo_nav",
                executable="twist_stamped_to_twist",
                name="cmu_cmd_to_cmd_vel",
                output="screen",
                parameters=[params],
                condition=_bool_condition("enable_cmd_vel_bridge"),
            ),
            Node(
                package="galileo_nav",
                executable="cmd_vel_mux",
                name="cmd_vel_mux",
                output="screen",
                parameters=[mux_params],
                condition=_bool_condition("enable_cmd_vel_bridge"),
            ),
        ]

    cmd_vel_bridge = OpaqueFunction(function=build_cmd_vel_bridge)

    return LaunchDescription(
        [
            mode_arg,
            kinematic_model_arg,
            global_path_topic_arg,
            localization_odom_topic_arg,
            cmu_input_scan_topic_arg,
            cmu_odom_topic_arg,
            enable_loam_interface_arg,
            enable_terrain_analysis_arg,
            enable_terrain_analysis_ext_arg,
            enable_sensor_scan_generation_arg,
            enable_cmu_local_planner_arg,
            local_cmu_config_file_arg,
            path_to_waypoint_config_file_arg,
            enable_path_to_waypoint_arg,
            final_goal_tolerance_arg,
            enable_cmu_start_gate_arg,
            enable_cmu_replan_arg,
            replan_cloud_topic_arg,
            map_name_arg,
            enable_cmd_vel_bridge_arg,
            cmu_cmd_vel_stamped_topic_arg,
            cmu_path_topic_arg,
            output_cmd_vel_topic_arg,
            set_ros_log_dir_to_launch_log_dir,
            TimerAction(period=1.0, actions=[loam_interface]),
            TimerAction(period=1.5, actions=[terrain_analysis]),
            TimerAction(period=1.5, actions=[terrain_analysis_ext]),
            TimerAction(period=1.5, actions=[sensor_scan_generation]),
            TimerAction(period=2.0, actions=[cmu_local_planner]),
            cmu_start_gate,
            path_tracking_nodes,
            cmd_vel_bridge,
            replan_monitor,
        ]
    )
