#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        # --- Subscribers ---
        self.create_subscription(Float32, '/perception/crosstrack_error', self.crosstrack_cb, 10)
        self.create_subscription(Float32, '/perception/heading_error', self.heading_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # --- Publisher ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- State Variables ---
        self.e = 0.0           # Crosstrack error (meters)
        self.th_e = 0.0        # Heading error (radians)
        self.v = 0.1           # Current speed (m/s)
        self.smooth_steer = 0.0  # Low-pass filtered steering angle

        # --- Stanley Tuning Parameters ---
        self.k = 1.5           # Crosstrack gain (increase = more aggressive correction)
        self.k_soft = 0.5      # Softening constant (stability at low speeds)
        self.target_speed = 1.0  # Driving speed (m/s)
        self.max_steer = 0.5     # Max steering angle (rad) — matches URDF joint limit
        self.alpha = 0.4         # Smoothing factor (0=heavy smooth, 1=no smooth)

        # --- Vehicle Geometry (from car.xacro) ---
        self.wheelbase = 0.28    # meters

        # Run control loop at 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Stanley Controller started.')

    def crosstrack_cb(self, msg):
        self.e = msg.data

    def heading_cb(self, msg):
        self.th_e = msg.data

    def odom_cb(self, msg):
        self.v = msg.twist.twist.linear.x
        if abs(self.v) < 0.1:
            self.v = 0.1  # Guard against division by zero

    def control_loop(self):
        # --- Stanley Steering Equation ---
        # delta = theta_e + arctan(k * e / (v + k_soft))
        atan_term = math.atan2(self.k * self.e, self.v + self.k_soft)
        steering_angle = self.th_e + atan_term

        # Clamp to physical limits
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        # Exponential smoothing to reduce jitter
        self.smooth_steer = self.alpha * steering_angle + (1.0 - self.alpha) * self.smooth_steer

        # --- KEY FIX: Convert steering angle → yaw rate ---
        # gazebo_ros_ackermann_drive interprets angular.z as a yaw rate (rad/s),
        # NOT a steering angle. Kinematic conversion:
        #   omega = v * tan(delta) / wheelbase
        yaw_rate = self.v * math.tan(self.smooth_steer) / self.wheelbase

        # Publish command
        cmd = Twist()
        cmd.linear.x = self.target_speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e={self.e:.3f}m  th_e={math.degrees(self.th_e):.1f}deg  '
            f'delta={math.degrees(self.smooth_steer):.1f}deg  '
            f'omega={yaw_rate:.3f}rad/s',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
