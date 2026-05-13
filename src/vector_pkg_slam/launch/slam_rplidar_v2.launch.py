from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    #pkg_rplidar = get_package_share_directory('rplidar_ros')
    pkg_my = get_package_share_directory('vector_pkg_slam')

    slam_params = os.path.join(pkg_my, 'config', 'slam.yaml')
    rviz_config = os.path.join(pkg_my, 'rviz', 'slam.rviz')
    
    """

    # ----------------------------
    # RPLIDAR
    # ----------------------------
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_rplidar, 'launch', 'rplidar_a2m12_launch.py')
        )
    )
    """

    # ----------------------------
    # TF: base_link -> laser
    # ----------------------------
    base_to_laser_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'laser']
    )

    # ----------------------------
    # DRIVE COMMAND NODE
    # ----------------------------
    Vehicle_Comm = Node(
        package='vector_pkg_slam',
        executable='Vehicle_Comm',
        name='Vehicle_Comm',
        output='screen'
    )
    
    drive_control = Node(
        package='vector_pkg_slam',
        executable='drive_control',
        name='drive_control',
        output='screen'
    )

    # ----------------------------
    # ODOM FROM KINEMATICS
    # ----------------------------
    vehicle_odom = Node(
        package='vector_pkg_slam',
        executable='vehicle_kin_odom',
        name='vehicle_kin_odom',
        output='screen'
    )

    # ----------------------------
    # SLAM TOOLBOX
    # ----------------------------
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params, {'use_sim_time': False}],
    )

    # ----------------------------
    # RVIZ
    # ----------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        additional_env={
            'LIBGL_ALWAYS_SOFTWARE': '1',
            'MESA_GL_VERSION_OVERRIDE': '3.3'
        },
        parameters=[{'use_sim_time': False}],
    )

    return LaunchDescription([
        #rplidar_launch,
        base_to_laser_tf,
        Vehicle_Comm,
        #drive_control,
        #vehicle_odom,
        slam_toolbox,
        rviz
    ])
