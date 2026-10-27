"""Simulación en Ignition Gazebo Fortress (ROS 2 Humble) con ros2_control y sensores simulados.

    export ROS_DOMAIN_ID=42
    ros2 launch robot_v3_description gazebo.launch.py                 # GUI de Gazebo + RViz
    ros2 launch robot_v3_description gazebo.launch.py headless:=true rviz:=false            # con pantalla/X
    ros2 launch robot_v3_description gazebo.launch.py headless:=true egl:=true rviz:=false  # por SSH, sin X

Tópicos ROS (puente ros_gz_bridge, config/gz_bridge.yaml), con los mismos nombres que el driver real de la D435
en la Jetson (camera_name=camera, namespace=camera) para poder probar el software de percepción/mapeo:
    /camera/camera/color/image_raw, /camera/camera/color/camera_info,
    /camera/camera/aligned_depth_to_color/image_raw, /camera/camera/depth/color/points, /scan, /clock
Base: /diff_drive_controller/cmd_vel_unstamped (geometry_msgs/Twist), odometría odom -> base_footprint.
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_v3_common import PACKAGE, check_domain, description_arguments, domain_guard_argument, robot_description  # noqa: E402,I100

CONTROLLERS = ['joint_state_broadcaster', 'arm_controller', 'gripper_controller', 'lift_controller', 'head_controller',
               'diff_drive_controller']


def resource_path(context):
    # Gazebo resuelve package://robot_v3_description/... buscando en <prefix>/share
    share = os.path.dirname(FindPackageShare(PACKAGE).perform(context))
    current = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    return [SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', os.pathsep.join(p for p in (share, current) if p))]


def generate_launch_description():
    pkg = FindPackageShare(PACKAGE)
    description = robot_description('gazebo')
    world = LaunchConfiguration('world')
    # headless: solo servidor (sin GUI). egl: renderizado sin X (cámara/LiDAR) por EGL, p. ej. por SSH en la NUC
    gz_args = PythonExpression(["'-r ' + ('-s ' if '", LaunchConfiguration('headless'), "' == 'true' else '')"
                                " + ('--headless-rendering ' if '", LaunchConfiguration('egl'), "' == 'true' else '') + '",
                                world, "'"])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={'gz_args': gz_args}.items())

    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-name', 'robot_v3', '-topic', 'robot_description', '-z', LaunchConfiguration('spawn_z')])

    # Al arrancar, Gazebo puede tardar en ejecutar ciclos de control (carga de sensores/render) y el cambio de estado
    # de un controlador expira a los 5 s: se espera hasta 60 s (--switch-timeout) y se activa primero
    # joint_state_broadcaster y después el resto, en un solo spawner.
    spawn_args = ['--controller-manager', '/controller_manager', '--controller-manager-timeout', '120',
                  '--switch-timeout', '60']
    spawn_jsb = Node(package='controller_manager', executable='spawner', output='screen',
                     arguments=[CONTROLLERS[0], *spawn_args])
    spawn_rest = Node(package='controller_manager', executable='spawner', output='screen',
                      arguments=[*CONTROLLERS[1:], *spawn_args])

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=PathJoinSubstitution([pkg, 'worlds', 'robot_v3_test.sdf'])),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false'],
                              description='Solo servidor de Gazebo (sin GUI)'),
        DeclareLaunchArgument('egl', default_value='false', choices=['true', 'false'],
                              description='Renderizado de sensores por EGL sin servidor X (--headless-rendering)'),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('spawn_z', default_value='0.01'),
        *description_arguments(),
        domain_guard_argument(),
        OpaqueFunction(function=check_domain),
        OpaqueFunction(function=resource_path),
        gazebo,
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': description, 'use_sim_time': True}]),
        spawn,
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[{'config_file': PathJoinSubstitution([pkg, 'config', 'gz_bridge.yaml']),
                          'use_sim_time': True}]),
        # Los controladores se cargan cuando el modelo (y su controller_manager) ya existe en Gazebo
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[spawn_jsb])),
        RegisterEventHandler(OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_rest])),
        Node(package='rviz2', executable='rviz2', output='screen', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', PathJoinSubstitution([pkg, 'rviz', 'robot_v3.rviz'])],
             parameters=[{'use_sim_time': True}]),
    ])
