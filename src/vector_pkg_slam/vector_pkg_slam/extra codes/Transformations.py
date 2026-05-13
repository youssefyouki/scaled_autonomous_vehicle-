import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

# Global broadcasters
_static_broadcaster = None
_dynamic_broadcaster = None

def init_tf_broadcasters(node):
    """Initialize static and dynamic TF broadcasters once."""
    global _static_broadcaster, _dynamic_broadcaster
    _static_broadcaster = StaticTransformBroadcaster(node)
    _dynamic_broadcaster = TransformBroadcaster(node)
    node.get_logger().info("TF broadcasters initialized.")


def publish_static_tf(node, frame_id, child_frame_id, translation, q):
    """
    Publish a static TF once with stamp=0.
    
    rotation_euler: tuple (roll, pitch, yaw) in radians
    """
    global _static_broadcaster
    if _static_broadcaster is None:
        node.get_logger().warn("Static broadcaster not initialized. Call init_tf_broadcasters() first.")
        return

    t = TransformStamped()
    t.header.stamp = rclpy.time.Time(seconds=0).to_msg()  # Time 0 for static TF
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    _static_broadcaster.sendTransform(t)
    node.get_logger().info(f"Published static TF: {frame_id} → {child_frame_id}")


def publish_dynamic_tf(node, frame_id, child_frame_id, stamp, translation, q):
    """
    Publish a dynamic TF continuously (e.g., odom → base_link).
    
    stamp: rclpy.time.Time or ROS2 msg timestamp
    rotation_euler: tuple (roll, pitch, yaw) in radians
    """
    global _dynamic_broadcaster
    if _dynamic_broadcaster is None:
        node.get_logger().warn("Dynamic broadcaster not initialized. Call init_tf_broadcasters() first.")
        return

    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    _dynamic_broadcaster.sendTransform(t)

