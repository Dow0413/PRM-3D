import os
from glob import glob

from setuptools import setup


package_name = "galileo_nav"

setup(
    name=package_name,
    version="0.0.0",
    package_dir={package_name: "src"},
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            "share/" + package_name,
            ["package.xml", "README.md"],
        ),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dow",
    maintainer_email="dow@todo.todo",
    description=(
        "Slim Galileo nav bringup: PRM multi-floor global planning with "
        "cmu/replan/wzh local planning (extracted standalone workspace)."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            # --- 局部规划节点 ---
            "local_replan = galileo_nav.local_replan:main",
            "cmu_replan = galileo_nav.cmu_replan:main",
            "wzh_replan = galileo_nav.wzh_replan:main",
            "path_follower = galileo_nav.path_follower:main",
            # --- CMU 循迹桥接链 ---
            "topo_path_to_waypoint = galileo_nav.topo_path_to_waypoint:main",
            "twist_stamped_to_twist = galileo_nav.twist_stamped_to_twist:main",
            "cmd_vel_mux = galileo_nav.cmd_vel_mux:main",
            "cmu_start_gate = galileo_nav.cmu_start_gate:main",
            "cmd_vel_postprocessor = galileo_nav.cmd_vel_postprocessor:main",
            # --- 辅助 ---
            "topic_state_logger = galileo_nav.topic_state_logger:main",
        ],
    },
)
