cmake_minimum_required(VERSION 3.8)
project(my_f1tenth_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install launch and model files so ros2 launch can discover them from share/.
install(DIRECTORY
  launch
  urdf
  worlds
  materials
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(
  PROGRAMS
    src/keyboard_drive.py
    src/camera_viewer.py
    src/velocity_keeper.py
    src/lane_detector_node.py
    src/stanley_control_node.py
    src/pure_pursuit_node.py
    src/Linderoth_control_node.py
    src/controller_panel.py
    src/rrt_planner_node.py
    src/astar_planner_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install LaneAssist library next to the scripts so lane_detector_node.py
# can locate it via __file__ at runtime (detect.py / lanekeeping.py / config.ini).
install(DIRECTORY
  src/laneassist
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
