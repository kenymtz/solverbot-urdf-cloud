"""Utilidades compartidas por los launch de robot_v3_description (no es un launch en sí)."""
import os

from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

PACKAGE = 'robot_v3_description'

# Dominio DDS del robot real (base_athome: ROS_DOMAIN_ID=10 en la NUC y la Jetson)
ROBOT_DOMAIN_ID = '10'


def description_arguments(camera_frames_default='true'):
    """Argumentos de la descripción comunes a todos los launch."""
    return [
        DeclareLaunchArgument('head_mount', default_value='column', choices=['column', 'lift'],
                              description='Cabezal sobre la columna fija (escaneo v2) o sobre el carro del elevador'),
        DeclareLaunchArgument('arm_limits', default_value='official', choices=['official', 'base_athome'],
                              description='Límites del Piper: oficiales de AgileX o los del URDF de base_athome'),
        DeclareLaunchArgument('camera_frames', default_value=camera_frames_default, choices=['true', 'false'],
                              description='Frames nominales de la D435 (false si el driver real publica su TF)'),
        DeclareLaunchArgument('wrist_camera', default_value='true', choices=['true', 'false'],
                              description='Frames de la cámara Orbbec de muñeca'),
    ]


def robot_description(ros2_control='none'):
    """robot_description como parámetro, generado con xacro a partir de los argumentos del launch."""
    xacro_file = PathJoinSubstitution([FindPackageShare(PACKAGE), 'urdf', 'robot_v3.urdf.xacro'])
    return ParameterValue(Command([
        'xacro ', xacro_file,
        ' ros2_control:=', ros2_control,
        ' head_mount:=', LaunchConfiguration('head_mount'),
        ' arm_limits:=', LaunchConfiguration('arm_limits'),
        ' camera_frames:=', LaunchConfiguration('camera_frames'),
        ' wrist_camera:=', LaunchConfiguration('wrist_camera'),
    ]), value_type=str)


def domain_guard_argument():
    return DeclareLaunchArgument(
        'allow_robot_domain', default_value='false', choices=['true', 'false'],
        description='Permitir estados SIMULADOS en el dominio DDS del robot real (ROS_DOMAIN_ID=10). No recomendado.')


def check_domain(context):
    """Evita mezclar estados simulados con el robot real: aborta si ROS_DOMAIN_ID es el del robot."""
    allow = LaunchConfiguration('allow_robot_domain').perform(context) == 'true'
    domain = os.environ.get('ROS_DOMAIN_ID', '0')
    if domain == ROBOT_DOMAIN_ID and not allow:
        raise RuntimeError(
            f'ROS_DOMAIN_ID={domain} es el dominio del robot real (NUC + Jetson). Este launch publica estados '
            'simulados (/joint_states, /tf) que se mezclarían con los reales. Use otro dominio, por ejemplo '
            '"export ROS_DOMAIN_ID=42", o allow_robot_domain:=true si sabe lo que hace.')
    return []
