"""TF del robot REAL a partir de los /joint_states que ya publica el robot. Solo LEE; no envía comandos.

Fuentes actuales de /joint_states (base_athome):
  * Jetson, piper_arm_controller.py: joint1..joint6, joint7, joint8 (brazo y pinza).
  * Jetson, pan_tilt_node: pan_joint, tilt_joint.
  * Nadie publica torso_lift_joint ni las ruedas (stepper_controller.py y hoverboard_driver no lo hacen).

joint_state_publisher combina /joint_states con valores por defecto para las articulaciones que nadie publica
(elevador y ruedas en 0) y publica el resultado en /robot_v3/joint_states, que es lo que lee
robot_state_publisher. Así no se reemite nada en /joint_states.

    ros2 launch robot_v3_description real_state.launch.py              # camera_frames:=false por defecto
    ros2 launch robot_v3_description real_state.launch.py rviz:=true

No lanzar junto con hoverboard_base/bringup.launch.py sin quitar su robot_state_publisher (TF duplicadas).
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_v3_common import PACKAGE, description_arguments, robot_description  # noqa: E402,I100

SOURCE_TOPIC = '/joint_states'          # estados reales (Jetson: brazo, pinza y pan-tilt)
MERGED_TOPIC = '/robot_v3/joint_states'


def generate_launch_description():
    description = robot_description()
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
        *description_arguments(camera_frames_default='false'),
        # En el namespace robot_v3 su salida relativa "joint_states" es /robot_v3/joint_states. No se usa un remapeo:
        # en ROS 2 también renombraría la fuente absoluta /joint_states y el nodo se escucharía a sí mismo.
        Node(package='joint_state_publisher', executable='joint_state_publisher', name='joint_state_merger',
             namespace='robot_v3', parameters=[{'source_list': [SOURCE_TOPIC], 'rate': 30}],
             remappings=[('robot_description', '/robot_description')]),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': description}],
             remappings=[('joint_states', MERGED_TOPIC)]),
        Node(package='rviz2', executable='rviz2', output='screen', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', PathJoinSubstitution([FindPackageShare(PACKAGE), 'rviz', 'robot_v3.rviz'])]),
    ])
