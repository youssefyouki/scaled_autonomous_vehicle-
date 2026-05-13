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
    # 1. Path to your URDF
    pkg_path = get_package_share_directory('my_f1tenth_sim')
    gazebo_pkg_path = get_package_share_directory('gazebo_ros')
    xacro_file = os.path.join(pkg_path, 'urdf', 'car.xacro')
    robot_description_config = xacro.process_file(xacro_file)

    gazebo_model_path = os.pathsep.join(
        [pkg_path, os.environ.get('GAZEBO_MODEL_PATH', '')]
    ).rstrip(os.pathsep)

    # Allow selecting a world; default to our two_lane_track.
    default_world = os.path.join(pkg_path, 'worlds', 'two_lane_track.world')
    declare_world_arg = DeclareLaunchArgument('world', default_value=default_world,
                                             description='Full path to world file to load')
    world = LaunchConfiguration('world')

    declare_gui_arg = DeclareLaunchArgument('gui', default_value='true',
                                            description='Set to false to run headless')
    gui = LaunchConfiguration('gui')

    declare_spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='0.0',
                                                description='Spawn X position in the world')
    declare_spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value='0.0',
                                                description='Spawn Y position in the world')
    declare_spawn_z_arg = DeclareLaunchArgument('spawn_z', default_value='0.15',
                                                description='Spawn Z position in the world')
    declare_spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='0.0',
                                                  description='Spawn yaw in radians')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    # 2. Start Gazebo server with the selected world and optionally the GUI client.
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg_path, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg_path, 'launch', 'gzclient.launch.py')
        ),
        condition=IfCondition(gui),
    )

    # Avoid long startup stalls when Gazebo tries to reach online model DB.
    disable_model_db = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')
    expose_package_models = SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path)
    
    # 3. Node to publish the robot state
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config.toxml()}]
    )

    # 4. Node to spawn the entity in Gazebo
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_f1tenth',
            '-timeout', '120',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-Y', spawn_yaw,
        ],
        output='screen'
    )

    # Give Gazebo a moment to initialize and advertise /spawn_entity.
    spawn_entity = TimerAction(
        period=5.0,
        actions=[spawn_entity_node]
    )

    return LaunchDescription([
        declare_world_arg,
        declare_gui_arg,
        declare_spawn_x_arg,
        declare_spawn_y_arg,
        declare_spawn_z_arg,
        declare_spawn_yaw_arg,
        disable_model_db,
        expose_package_models,
        gzserver,
        gzclient,
        node_robot_state_publisher,
        spawn_entity,
    ])
