"""RViz con estados articulares SIMULADOS (joint_state_publisher[_gui]). No se conecta a hardware.

    ros2 launch robot_v3_description display.launch.py                 # sliders
    ros2 launch robot_v3_description display.launch.py gui:=false      # todo en cero (o en el centro del rango)
    opciones: head_mount:=column|lift  arm_limits:=official|base_athome  camera_frames:=true|false  rviz:=false
              arm_pose:=home|zero   pose inicial del brazo (home = DEFAULT_HOME de la Jetson, la del escaneo)
"""
import math
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


# DEFAULT_HOME de piper_arm_controller.py (Jetson), en grados: el brazo plegado al costado de la caja
DEFAULT_HOME_DEG = [86.28, 173.91, -164.0, 66.98, 0.69, -146.41]


def initial_state(context):
    """Valores iniciales de joint_state_publisher[_gui] (parámetro "zeros")."""
    if LaunchConfiguration('arm_pose').perform(context) == 'home':
        return {f'joint{i + 1}': math.radians(v) for i, v in enumerate(DEFAULT_HOME_DEG)}
    return {}


def state_publishers(context):
    params = [{'zeros': initial_state(context)}] if initial_state(context) else []
    gui = LaunchConfiguration('gui')
    return [
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui', parameters=params,
             condition=IfCondition(gui)),
        Node(package='joint_state_publisher', executable='joint_state_publisher', parameters=params,
             condition=UnlessCondition(gui)),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false'],
                              description='joint_state_publisher_gui (sliders) o joint_state_publisher'),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('arm_pose', default_value='home', choices=['home', 'zero'],
                              description='Pose inicial del brazo en los sliders'),
        *description_arguments(),
        domain_guard_argument(),
        OpaqueFunction(function=check_domain),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_description()}]),
        OpaqueFunction(function=state_publishers),
        Node(package='rviz2', executable='rviz2', output='screen', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', PathJoinSubstitution([FindPackageShare(PACKAGE), 'rviz', 'robot_v3.rviz'])]),
    ])
