"""SolverBot v3 en Gazebo Harmonic (ROS 2 Jazzy): arena doméstica o mundo de entrenamiento de Pick and Place.

    export ROS_DOMAIN_ID=42                      # NUNCA el dominio del robot real (10)
    ros2 launch solverbot_sim sim.launch.py                                   # arena + GUI + ambas cámaras
    ros2 launch solverbot_sim sim.launch.py world:=pick_place_training        # mundo de entrenamiento
    ros2 launch solverbot_sim sim.launch.py world:=pick_place_training headless:=true cameras:=false \
        camera_profile:=train                                                 # modo RL (lo usa solverbot_rl)
El mundo arranca corriendo (los spawners necesitan ciclos de control); el entorno RL lo pausa al conectarse y luego
avanza el tiempo con /world/<mundo>/control (multi_step).

Tópicos ROS: /camera/camera/color/image_raw, /camera/camera/aligned_depth_to_color/image_raw,
/camera/camera/color/camera_info, /wrist_camera/image_raw, /wrist_camera/camera_info, /scan, /clock, /joint_states.
Servicios (puente): /world/<mundo>/control (ros_gz_interfaces/ControlWorld), /world/<mundo>/set_pose.
Pose real del objeto de entrenamiento (solo recompensas/métricas, NO observación): /model/pick_object/pose.
Controladores (controllers:=rl): arm/gripper/lift/head_position_controller (forward_command) + diff_drive_controller.
"""
import os
import sys

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

ROBOT_DOMAIN_ID = '10'   # dominio DDS del robot real (base_athome)
CONTROLLERS = {
    'rl': ['arm_position_controller', 'gripper_position_controller', 'lift_position_controller',
           'head_position_controller', 'diff_drive_controller'],
    'trajectory': ['arm_controller', 'gripper_controller', 'lift_controller', 'head_controller',
                   'diff_drive_controller'],
}


def launch_setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in (
        'world', 'headless', 'egl', 'cameras', 'rviz', 'camera_profile', 'controllers', 'x', 'y', 'yaw',
        'allow_robot_domain', 'gz_partition')}
    domain = os.environ.get('ROS_DOMAIN_ID', '0')
    if domain == ROBOT_DOMAIN_ID and a['allow_robot_domain'] != 'true':
        raise RuntimeError(f'ROS_DOMAIN_ID={domain} es el dominio del robot real. Use otro, p. ej. export ROS_DOMAIN_ID=42')

    sim_share = FindPackageShare('solverbot_sim').perform(context)
    robot_share = FindPackageShare('robot_v3_description').perform(context)
    world_file = a['world'] if a['world'].endswith('.sdf') else os.path.join(sim_share, 'worlds', a['world'] + '.sdf')
    world_name = os.path.splitext(os.path.basename(world_file))[0]

    # Pose de aparición por defecto según el mundo
    x, y, yaw = a['x'], a['y'], a['yaw']
    if x == '' and world_name == 'home_arena':
        with open(os.path.join(sim_share, 'config', 'arena_spec.yaml')) as fh:
            st = yaml.safe_load(fh)['start_locations']['pick_place_demo']
        x, y, yaw = str(st['xy'][0]), str(st['xy'][1]), str(st['yaw'])
    x, y, yaw = x or '0.0', y or '0.0', yaw or '0.0'

    controllers_yaml = os.path.join(sim_share, 'config', 'controllers_rl.yaml') if a['controllers'] == 'rl' else \
        os.path.join(robot_share, 'config', 'ros2_controllers.yaml')
    description = ParameterValue(Command([
        'xacro ', os.path.join(robot_share, 'urdf', 'robot_v3.urdf.xacro'), ' ros2_control:=gz',
        ' camera_profile:=', a['camera_profile'], ' controllers:=', controllers_yaml]), value_type=str)

    gz_args = '-r ' + ('-s ' if a['headless'] == 'true' else '') + \
        ('--headless-rendering ' if a['egl'] == 'true' else '') + world_file
    resource = os.pathsep.join(p for p in (os.path.dirname(robot_share), os.environ.get('GZ_SIM_RESOURCE_PATH', '')) if p)

    spawn_args = ['--controller-manager', '/controller_manager', '--controller-manager-timeout', '120',
                  '--switch-timeout', '60']
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-name', 'robot_v3', '-topic', 'robot_description', '-x', x, '-y', y, '-z', '0.01',
                            '-Y', yaw])
    spawn_jsb = Node(package='controller_manager', executable='spawner', output='screen',
                     arguments=['joint_state_broadcaster', *spawn_args])
    spawn_rest = Node(package='controller_manager', executable='spawner', output='screen',
                      arguments=[*CONTROLLERS[a['controllers']], *spawn_args])
    w = f'/world/{world_name}'
    # Pose real de los objetos con PosePublisher (solo recompensas/métricas, nunca observación de la política)
    with open(os.path.join(sim_share, 'config', 'pick_place.yaml')) as fh:
        tracked = ['robot_v3'] + ([yaml.safe_load(fh)['world']['object']['name']]
                                  if world_name == 'pick_place_training' else [])
    actions = [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(FindPackageShare('ros_gz_sim').perform(context), 'launch',
                                                       'gz_sim.launch.py')),
            launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items()),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='log',
             parameters=[{'robot_description': description, 'use_sim_time': True}]),
        spawn,
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='sensor_bridge', output='log',
             parameters=[{'config_file': os.path.join(sim_share, 'config', 'gz_bridge_sensors.yaml'),
                          'use_sim_time': True}]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='world_bridge', output='log',
             arguments=[f'{w}/control@ros_gz_interfaces/srv/ControlWorld',
                        f'{w}/set_pose@ros_gz_interfaces/srv/SetEntityPose',
                        *[f'/model/{o}/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V' for o in tracked]],
             parameters=[{'use_sim_time': True}]),
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[spawn_jsb])),
        RegisterEventHandler(OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_rest])),
    ]
    if a['cameras'] == 'true':
        # Visualización simultánea de ambas cámaras durante una demostración
        for name, topic in (('head_camera_view', '/camera/camera/color/image_raw'),
                            ('wrist_camera_view', '/wrist_camera/image_raw')):
            actions.append(Node(package='rqt_image_view', executable='rqt_image_view', name=name, output='log',
                                arguments=[topic], parameters=[{'use_sim_time': True}]))
    if a['rviz'] == 'true':
        actions.append(Node(package='rviz2', executable='rviz2', output='log',
                            arguments=['-d', os.path.join(robot_share, 'rviz', 'robot_v3.rviz')],
                            parameters=[{'use_sim_time': True}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='home_arena',
                              description='home_arena | pick_place_training | ruta a un .sdf'),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('egl', default_value='false', choices=['true', 'false'],
                              description='Render de sensores por EGL sin servidor X (--headless-rendering)'),
        DeclareLaunchArgument('cameras', default_value='true', choices=['true', 'false'],
                              description='Abrir rqt_image_view para la cámara de cabeza y la de la pinza'),
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('camera_profile', default_value='demo', choices=['demo', 'train']),
        DeclareLaunchArgument('controllers', default_value='rl', choices=['rl', 'trajectory']),
        DeclareLaunchArgument('x', default_value=''),
        DeclareLaunchArgument('y', default_value=''),
        DeclareLaunchArgument('yaw', default_value=''),
        DeclareLaunchArgument('allow_robot_domain', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('gz_partition', default_value='',
                              description='(informativo) exportar GZ_PARTITION antes del launch para aislar instancias'),
        OpaqueFunction(function=launch_setup),
    ])


if __name__ == '__main__':
    sys.exit('Usar: ros2 launch solverbot_sim sim.launch.py')
