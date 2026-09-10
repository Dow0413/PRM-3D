from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='loam_interface',
            executable='loamInterface',
            name='loamInterface',
            output='screen',
            parameters=[{
                'stateEstimationTopic': '/integrated_to_init',
                'registeredScanTopic': '/velodyne_cloud_registered',
                'flipStateEstimation': True,
                'flipRegisteredScan': True,
                'sendTF': True,
                'reverseTF': False,
            }],
        )
    ])
