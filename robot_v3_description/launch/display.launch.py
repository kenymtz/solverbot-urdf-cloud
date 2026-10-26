"""RViz con estados articulares SIMULADOS (joint_state_publisher[_gui]). No se conecta a hardware.

    ros2 launch robot_v3_description display.launch.py                 # sliders
    ros2 launch robot_v3_description display.launch.py gui:=false      # todo en cero (o en el centro del rango)
    opciones: head_mount:=column|lift  arm_limits:=official|base_athome  camera_frames:=true|false  rviz:=false
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_v3_common import PACKAGE, check_domain, description_arguments, domain_guard_argument, robot_description  # noqa: E402,I100


def generate_launch_description():
    gui = LaunchConfiguration('gui')
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false'],
                              description='joint_state_publisher_gui (sliders) o joint_state_publisher'),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        *description_arguments(),
        domain_guard_argument(),
        OpaqueFunction(function=check_domain),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_description()}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui', condition=IfCondition(gui)),
        Node(package='joint_state_publisher', executable='joint_state_publisher', condition=UnlessCondition(gui)),
        Node(package='rviz2', executable='rviz2', output='screen', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', PathJoinSubstitution([FindPackageShare(PACKAGE), 'rviz', 'robot_v3.rviz'])]),
    ])
