"""
astar_sim.launch.py
====================
All-in-one launch for A* path-planning mode:
  1. Gazebo + two_lane_track world + car spawn
  2. A* planner node           (t = 12 s — waits for Gazebo + odometry to be live)
  3. Controller panel GUI

The lane detector is intentionally NOT started — the A* planner publishes
CTE/heading directly on /perception/crosstrack_error and /perception/heading_error
so any controller selected in the panel follows the planned path.

Usage:
  ros2 launch my_f1tenth_sim astar_sim.launch.py
  ros2 launch my_f1tenth_sim astar_sim.launch.py gui:=false   # headless Gazebo

Then in RViz2:
  - Fixed Frame = odom
  - Add → By topic → /planned_path   (Path)
  - Add → By topic → /astar/markers  (MarkerArray)
  - Add → By topic → /astar/map      (Map)
  - Click "2D Goal Pose" to send a navigation goal
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_path = get_package_share_directory('my_f1tenth_sim')

    declare_gui_arg = DeclareLaunchArgument(
        'gui', default_value='true',
        description='Set to false to run Gazebo headless')

    # ── 1. Gazebo + car spawn ─────────────────────────────────────────────────
    spawn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_path, 'launch', 'spawn_car.launch.py')
        ),
        launch_arguments={
            'gui':       'true',
            'spawn_x':   '2.2668',
            'spawn_y':   '-0.1515',
            'spawn_z':   '0.15',
            'spawn_yaw': '-1.5293',
        }.items(),
    )

    # ── 2. A* planner — wait for Gazebo + odometry to be ready ───────────────
    astar_planner = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='my_f1tenth_sim',
                executable='astar_planner_node.py',
                name='astar_planner',
                output='screen',
            )
        ],
    )

    # ── 3. Controller panel GUI ───────────────────────────────────────────────
    controller_panel = Node(
        package='my_f1tenth_sim',
        executable='controller_panel.py',
        name='controller_panel',
        output='screen',
    )

    # ── 4. RViz2 pre-configured for A* ────────────────────────────────────────
    rviz_config = os.path.join(pkg_path, 'config', 'astar_rviz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([
        declare_gui_arg,
        spawn_launch,
        astar_planner,
        controller_panel,
        rviz,
    ])
