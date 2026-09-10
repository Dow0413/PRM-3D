import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory("local_cloud_filter")
    default_params_file = os.path.join(
        package_share_dir, "config", "local_cloud_filter_jt128.yaml"
    )

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="Path to the local cloud filter parameter file.",
            ),
            Node(
                package="local_cloud_filter",
                executable="local_cloud_filter_node",
                name="local_cloud_filter_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
