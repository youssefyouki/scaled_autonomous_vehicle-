#!/usr/bin/env python3
"""
Velocity keeper node.
Publishes zero velocity on /cmd_vel to keep the car stationary when no command is active.
This counteracts garbage initialization from gazebo_ros_ackermann_drive.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityKeeper(Node):
    def __init__(self):
        super().__init__('velocity_keeper')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.publish_zero_velocity)
        self.get_logger().info('Velocity keeper started: publishing zero velocity.')

    def publish_zero_velocity(self):
        """Publish a zero velocity command to keep the car still."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityKeeper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
