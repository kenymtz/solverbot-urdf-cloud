"""Simulación SIN física ni hardware: ros2_control con mock_components/GenericSystem.

Permite comandar elevador, brazo, pinza y cabezal con controladores reales de ROS 2 (joint_trajectory_controller)
y verlos en RViz. No abre puertos serie, CAN ni USB.

    export ROS_DOMAIN_ID=42      # cualquier dominio distinto del robot (10)
    ros2 launch robot_v3_description sim_mock.launch.py
    ros2 run robot_v3_description demo_motion.py          # secuencia de prueba
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_v3_common import PACKAGE, check_domain, description_arguments, domain_guard_argument, robot_description  # noqa: E402,I100

CONTROLLERS = ['joint_state_broadcaster', 'arm_controller', 'gripper_controller', 'lift_controller', 'head_controller']


def generate_launch_description():
    description = robot_description('mock')
    controllers_yaml = PathJoinSubstitution([FindPackageShare(PACKAGE), 'config', 'ros2_controllers.yaml'])
    spawners = [
        Node(package='controller_manager', executable='spawner', output='screen',
             arguments=[name, '--controller-manager', '/controller_manager', '--controller-manager-timeout', '60'])
        for name in CONTROLLERS
    ]
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        *description_arguments(),
        domain_guard_argument(),
        OpaqueFunction(function=check_domain),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': description}]),
        Node(package='controller_manager', executable='ros2_control_node', output='screen',
             parameters=[{'robot_description': description}, controllers_yaml]),
        *spawners,
        Node(package='rviz2', executable='rviz2', output='screen', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', PathJoinSubstitution([FindPackageShare(PACKAGE), 'rviz', 'robot_v3.rviz'])]),
    ])
