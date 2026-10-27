#!/bin/bash
# Dependencias para WSL2 + Ubuntu 24.04 + ROS 2 Jazzy (ya instalado en /opt/ros/jazzy).
# Instala SOLO lo necesario para la simulación (sin drivers de hardware) y un venv de Python para RL.
set -e
sudo apt-get update
sudo apt-get install -y \
  ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-rqt-image-view ros-jazzy-rviz2 \
  python3-venv python3-colcon-common-extensions python3-scipy python3-yaml
# venv con acceso a rclpy del sistema (--system-site-packages). PyTorch CPU (~200 MB): no hace falta CUDA para MLP;
# si tenés GPU NVIDIA con CUDA en WSL podés instalar la variante CUDA en su lugar.
python3 -m venv --system-site-packages ~/solverbot_venv
source ~/solverbot_venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "gymnasium>=0.29" "stable-baselines3>=2.3" pytest
echo "Listo. Activar con: source ~/solverbot_venv/bin/activate"
