"""
full_sim.launch.py
==================
One-shot launch:
  1. Gazebo + two_lane_track world  (via spawn_with_track.launch.py)
  2. Lane detector node             (t = 10 s — camera needs the car in world)
  3. Controller Panel GUI           (t = 12 s — after lane detector is up)

The controller panel opens a tkinter window for selecting and tuning
controllers.  The lane detector opens its own cv2 windows (LaneAssist peaks,
CLAHE gray, BEV warped).

Usage:
  ros2 launch my_f1tenth_sim full_sim.launch.py
  ros2 launch my_f1tenth_sim full_sim.launch.py gui:=false   # headless Gazebo
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

    # ── 2. Lane detector — wait for car + camera to be live ───────────────────
    lane_detector = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='my_f1tenth_sim',
                executable='lane_detector_node.py',
                name='lane_detector',
                output='screen',
            )
        ],
    )

    # ── 3. Controller panel GUI ───────────────────────────────────────────────
    controller_panel = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='my_f1tenth_sim',
                executable='controller_panel.py',
                name='controller_panel',
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        declare_gui_arg,
        spawn_launch,
        lane_detector,
        controller_panel,
    ])
