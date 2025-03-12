
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="cev_offtrack_planner_ros2",
                executable="planner_node",
                name="cev_offtrack_planner_ros2",
                output="screen",
            ),
        ]
    )

