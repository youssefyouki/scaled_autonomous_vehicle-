import rclpy
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
import numpy as np
import struct
from controller import Lidar
import math

def publish_scan(node, lidar, curr_time):
    # Get Lidar properties
    fov = lidar.getFov()              # radians
    num_points = lidar.getHorizontalResolution()
    
    # Prepare LaserScan message
    msg = LaserScan()
    msg.header.stamp = curr_time
    msg.header.frame_id = 'lidar_link'

    # Correct angle calculation
    print("Lidar FOV:", fov, "radians =", math.degrees(fov), "degrees")

    msg.angle_min = -fov / 2.0
    msg.angle_max = fov / 2.0
    msg.angle_increment = (msg.angle_max - msg.angle_min) / (num_points - 1)

    msg.range_min = lidar.getMinRange()
    msg.range_max = lidar.getMaxRange()

    # Initialize ranges with max range
    ranges = [msg.range_max] * num_points

    # Project each 3D point into 2D plane and compute polar angle
    point_cloud = lidar.getPointCloud()
    for p in point_cloud:
        x, y = p.x, p.y  # Webots: x forward, y left, z up
        r = math.sqrt(x*x + y*y)
        theta = math.atan2(y, x)
        idx = int(round((theta - msg.angle_min) / msg.angle_increment))
        if 0 <= idx < num_points:
            ranges[idx] = min(r, ranges[idx])

    msg.ranges = ranges
    return msg

def publish_pointcloud(node, lidar, curr_time):
    """
    Convert Webots Velodyne HDL-32E point cloud to ROS2 PointCloud2 message.
    Each Webots LiDAR 'getPointCloud()' returns 3D points for all 32 channels.
    """
    pc = lidar.getPointCloud()
    num_points = len(pc)

    msg = PointCloud2()
    msg.header.stamp = curr_time
    msg.header.frame_id = 'lidar_link'  # must match TF and SLAM settings

    msg.height = 1  # unorganized cloud
    msg.width = num_points
    msg.is_bigendian = False
    msg.is_dense = True

    # Define x, y, z fields (float32)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.point_step = 12  # 3 * 4 bytes
    msg.row_step = msg.point_step * num_points

    # Build binary data buffer
    buffer = bytearray()
    for p in pc:
        # Webots axes: X forward, Y left, Z up — ROS uses same convention
        buffer.extend(struct.pack('fff', float(p.x), float(p.y), float(p.z)))

    msg.data = bytes(buffer)
    return msg

