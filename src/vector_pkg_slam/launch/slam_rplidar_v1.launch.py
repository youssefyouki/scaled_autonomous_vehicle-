from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime


def generate_launch_description():

    # ----------------------------
    # Package paths
    # ----------------------------
    pkg_rplidar = get_package_share_directory('rplidar_ros')
    pkg_my = get_package_share_directory('vector_pkg_slam')

    slam_params = os.path.join(pkg_my, 'config', 'slam.yaml')
    rviz_config = os.path.join(pkg_my, 'rviz', 'slam.rviz')

    # ----------------------------
    # Timestamped map directory
    # ----------------------------
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
     maps_dir = os.path.join("~/ros2_ws/src/",pkg_my, 'maps', now)
    except:
     print("FNF")
    # ----------------------------
    # RPLIDAR
    # ----------------------------
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_rplidar, 'launch', 'rplidar_a2m12_launch.py')
        )
    )

    # ----------------------------
    # TF: base_link → laser
    # ----------------------------
    base_to_laser_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'laser']
    )

    # ----------------------------
    # TF: odom → base_link (FAKE ODOM)
    # ----------------------------
    odom_to_base_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_base_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
    )
    
    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    # ----------------------------
    # SLAM Toolbox
    # ----------------------------
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params]
    )

    # ----------------------------
    # RViz (GPU SAFE)
    # ----------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        additional_env={
            'LIBGL_ALWAYS_SOFTWARE': '1',
            'MESA_GL_VERSION_OVERRIDE': '3.3'
        }
    )

    # ----------------------------
    # AUTO SAVE MAP ON SHUTDOWN (timestamped)
    # ----------------------------
    save_map_on_shutdown = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                ExecuteProcess(
                    cmd=[
                        'bash', '-c',
                        f'mkdir -p {maps_dir} && sleep 2 && ros2 run nav2_map_server map_saver_cli -f {maps_dir}/slam_map'
                    ],
                    output='screen'
                )
            ]
        )
    )

    # ----------------------------
    # LAUNCH DESCRIPTION
    # ----------------------------
    return LaunchDescription([
        rplidar_launch,
        map_to_odom_tf,
        base_to_laser_tf,
        odom_to_base_tf,
        slam_toolbox,
        rviz,
        save_map_on_shutdown
    ])
