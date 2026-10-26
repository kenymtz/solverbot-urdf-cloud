"""Visualización en RViz con estados de articulación SIMULADOS. No se conecta a hardware ni publica comandos."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('robot_v2_description')
    robot_description = ParameterValue(Command([
        'xacro ', PathJoinSubstitution([pkg, 'urdf', 'robot_v2.urdf.xacro']),
        ' use_estimated_inertials:=', LaunchConfiguration('use_estimated_inertials'),
        ' with_camera_optical_frame:=', LaunchConfiguration('with_camera_optical_frame')]), value_type=str)
    gui = LaunchConfiguration('gui')
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='joint_state_publisher_gui (sliders) o joint_state_publisher (ceros)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('use_estimated_inertials', default_value='false'),
        DeclareLaunchArgument('with_camera_optical_frame', default_value='true'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}], output='screen'),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui', condition=IfCondition(gui)),
        Node(package='joint_state_publisher', executable='joint_state_publisher', condition=UnlessCondition(gui)),
        Node(package='rviz2', executable='rviz2', output='screen', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', PathJoinSubstitution([pkg, 'rviz', 'display.rviz'])]),
    ])
