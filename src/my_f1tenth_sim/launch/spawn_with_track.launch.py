import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_path = get_package_share_directory('my_f1tenth_sim')
    gazebo_pkg_path = get_package_share_directory('gazebo_ros')
    xacro_file = os.path.join(pkg_path, 'urdf', 'car.xacro')
    robot_description_config = xacro.process_file(xacro_file)

    # Point gazebo to our world file inside the package
    world_file = os.path.join(pkg_path, 'worlds', 'two_lane_track.world')

    declare_gui_arg = DeclareLaunchArgument('gui', default_value='true',
                                            description='Set to false to run headless')
    gui = LaunchConfiguration('gui')

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg_path, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg_path, 'launch', 'gzclient.launch.py')
        ),
        condition=IfCondition(gui)
    )

    disable_model_db = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config.toxml()}]
    )

    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_f1tenth', '-timeout', '120', '-z', '0.05'],
        output='screen'
    )

    spawn_entity = TimerAction(
        period=5.0,
        actions=[spawn_entity_node]
    )

    return LaunchDescription([
        declare_gui_arg,
        disable_model_db,
        gazebo_server,
        gazebo_client,
        node_robot_state_publisher,
        spawn_entity,
    ])
