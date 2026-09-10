import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    sensor_offset_x = LaunchConfiguration('sensorOffsetX')
    sensor_offset_y = LaunchConfiguration('sensorOffsetY')
    camera_offset_z = LaunchConfiguration('cameraOffsetZ')
    two_way_drive = LaunchConfiguration('twoWayDrive')
    max_speed = LaunchConfiguration('maxSpeed')
    autonomy_mode = LaunchConfiguration('autonomyMode')
    autonomy_speed = LaunchConfiguration('autonomySpeed')
    joy_to_speed_delay = LaunchConfiguration('joyToSpeedDelay')
    goal_x = LaunchConfiguration('goalX')
    goal_y = LaunchConfiguration('goalY')
    path_folder = LaunchConfiguration('pathFolder')
    search_radius = LaunchConfiguration('searchRadius')
    use_path_hysteresis = LaunchConfiguration('usePathHysteresis')
    path_hysteresis_score_ratio = LaunchConfiguration('pathHysteresisScoreRatio')

    return LaunchDescription([
        DeclareLaunchArgument('sensorOffsetX', default_value='0.0', description=''),
        DeclareLaunchArgument('sensorOffsetY', default_value='0.0', description=''),
        DeclareLaunchArgument('cameraOffsetZ', default_value='0.0', description=''),
        DeclareLaunchArgument('twoWayDrive', default_value='false', description=''), # true: 倒车 
        DeclareLaunchArgument('maxSpeed', default_value='2.0', description=''),
        DeclareLaunchArgument('autonomyMode', default_value='true', description=''),
        DeclareLaunchArgument('autonomySpeed', default_value='2.0', description=''),
        DeclareLaunchArgument('joyToSpeedDelay', default_value='2.0', description=''),
        DeclareLaunchArgument('goalX', default_value='0.0', description=''),
        DeclareLaunchArgument('goalY', default_value='0.0', description=''),
        DeclareLaunchArgument('usePathHysteresis', default_value='true', description=''),
        DeclareLaunchArgument('pathHysteresisScoreRatio', default_value='0.85', description=''),
        DeclareLaunchArgument(
            'pathFolder',
            default_value=os.path.join(get_package_share_directory('local_planner'), 'paths', '0_45'),
            description='Directory containing startPaths.ply, paths.ply, pathList.ply, and correspondences.txt',
        ),
        DeclareLaunchArgument('searchRadius', default_value='0.45', description='Path collision inflation radius'),
        Node(
            package='local_planner',
            executable='localPlanner',
            name='localPlanner',
            output='screen',
            parameters=[{
                'pathFolder': path_folder,
                'vehicleLength': 0.5,
                'vehicleWidth': 0.5,
                'sensorOffsetX': sensor_offset_x,
                'sensorOffsetY': sensor_offset_y,
                'twoWayDrive': two_way_drive,
                'laserVoxelSize': 0.05,
                'terrainVoxelSize': 0.1,  # 0.1
                'useTerrainAnalysis': True,
                'checkObstacle': True,
                'checkRotObstacle': False,
                'adjacentRange': 4.25,
                'searchRadius': search_radius,
                'obstacleHeightThre': 0.45,
                'groundHeightThre': 0.1,
                'costHeightThre': 0.1,
                'costScore': 0.02,
                'useCost': False,
                'pointPerPathThre': 2,  # 8
                'minRelZ': -0.5,
                'maxRelZ': 0.6,  # 扫描区域相对于机器人高度 0.25 260604
                'maxSpeed': max_speed,
                'dirWeight': 0.02,
                'dirThre': 90.0,
                'dirToVehicle': False,
                'pathScale': 1.25,
                'minPathScale': 0.75,
                'pathScaleStep': 0.25,
                'pathScaleBySpeed': True,
                'minPathRange': 1.0,
                'pathRangeStep': 0.5,
                'pathRangeBySpeed': True,
                'pathCropByGoal': True,
                'usePathHysteresis': use_path_hysteresis,
                'pathHysteresisScoreRatio': path_hysteresis_score_ratio,
                'autonomyMode': autonomy_mode,
                'autonomySpeed': autonomy_speed,
                'joyToSpeedDelay': joy_to_speed_delay,
                'joyToCheckObstacleDelay': 5.0,
                'goalClearRange': 0.5,
                'goalX': goal_x,
                'goalY': goal_y,
            }],
        ),
        Node(
            package='local_planner',
            executable='pathFollower',
            name='pathFollower',
            output='screen',
            parameters=[{
                'sensorOffsetX': sensor_offset_x,
                'sensorOffsetY': sensor_offset_y,
                'pubSkipNum': 1,
                'twoWayDrive': two_way_drive,
                'lookAheadDis': 0.8,  #1.0
                'yawRateGain': 3.0,
                'stopYawRateGain': 3.0,
                'maxYawRate': 0.523599,
                'maxSpeed': max_speed,
                'maxAccel': 0.6,
                'switchTimeThre': 1.0,
                'dirDiffThre': 0.2,
                'stopDisThre': 0.25,
                'slowDwnDisThre': 0.85,
                'useInclRateToSlow': False,
                'inclRateThre': 120.0,
                'slowRate1': 0.25,
                'slowRate2': 0.5,
                'slowTime1': 2.0,
                'slowTime2': 2.0,
                'useInclToStop': False,
                'inclThre': 45.0,
                'stopTime': 5.0,
                'noRotAtStop': False,
                'noRotAtGoal': True,
                'autonomyMode': autonomy_mode,
                'autonomySpeed': autonomy_speed,
                'joyToSpeedDelay': joy_to_speed_delay,
            }],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='vehicleTransPublisher',
            arguments=[
                PythonExpression(['-1.0 * ', sensor_offset_x]),
                PythonExpression(['-1.0 * ', sensor_offset_y]),
                '0',
                '0',
                '0',
                '0',
                '/sensor',
                '/vehicle',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='sensorTransPublisher',
            arguments=[
                '0',
                '0',
                camera_offset_z,
                '-1.5707963',
                '0',
                '-1.5707963',
                '/sensor',
                '/camera',
            ],
        ),
    ])
