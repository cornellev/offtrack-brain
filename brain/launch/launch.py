from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def get_path(package, dir, file):
    return os.path.join(
        get_package_share_directory(package),
        dir,
        file
    )

def launch(package, file, launch_folder="launch", arguments={}):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            get_path(package, launch_folder, file)
        ),
        launch_arguments=arguments.items()
    )

def generate_launch_description():
    return LaunchDescription(
        []
    )
