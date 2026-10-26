cmake_minimum_required(VERSION 3.10)
project(robot_v3_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch meshes rviz urdf worlds DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/demo_motion.py tools/validate_v3.py tools/runtime_checks.py tools/check_real_state.py
        DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Validación estática (xacro, árbol, límites, cinemática, frames, recursos) sobre el paquete instalado
  ament_add_pytest_test(test_robot_v3_description test/test_robot_v3_description.py TIMEOUT 300)
endif()

ament_package()
