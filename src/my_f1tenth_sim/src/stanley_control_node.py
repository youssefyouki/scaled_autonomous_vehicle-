#!/usr/bin/env python3
"""
Stanley Lane-Keeping Controller
================================
Standard Stanley lateral control law.

Subscribes:
  /perception/crosstrack_error  (std_msgs/Float32)  lateral offset from lane centre (m)
  /perception/heading_error     (std_msgs/Float32)  heading difference (rad)
  /odom                         (nav_msgs/Odometry)  vehicle speed

Publishes:
  /cmd_vel  (geometry_msgs/Twist)

Stanley law:
  delta = theta_e + atan(k * e / (v + k_soft))
  omega = v * tan(delta) / L
"""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult


class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        self.create_subscription(Float32, '/perception/crosstrack_error',
                                 self.crosstrack_cb, 10)
        self.create_subscription(Float32, '/perception/heading_error',
                                 self.heading_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.e            = 0.0
        self.th_e         = 0.0
        self.v_actual     = 0.1
        self.smooth_steer = 0.0

        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S  = 1.0

        self.target_speed = 0.5    # m/s
        self.k            = 1.0    # crosstrack gain
        self.k_soft       = 0.4    # softening term — prevents div/0 at v≈0
        self.alpha        = 0.5    # EMA smoothing weight
        self.max_steer    = 0.5    # rad
        self.wheelbase    = 0.28   # m

        self.declare_parameter('target_speed', self.target_speed)
        self.declare_parameter('k',            self.k)
        self.declare_parameter('k_soft',       self.k_soft)
        self.declare_parameter('alpha',        self.alpha)
        self.add_on_set_parameters_callback(self._on_param_change)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Stanley Controller started.')

    def _on_param_change(self, params):
        for p in params:
            if   p.name == 'target_speed': self.target_speed = p.value
            elif p.name == 'k':            self.k            = p.value
            elif p.name == 'k_soft':       self.k_soft       = p.value
            elif p.name == 'alpha':        self.alpha        = p.value
        return SetParametersResult(successful=True)

    def crosstrack_cb(self, msg: Float32):
        self.e = msg.data
        self.last_perception_stamp = self.get_clock().now()

    def heading_cb(self, msg: Float32):
        self.th_e = max(-math.radians(20.0), min(math.radians(20.0), msg.data))

    def odom_cb(self, msg: Odometry):
        self.v_actual = max(0.05, abs(msg.twist.twist.linear.x))

    def control_loop(self):
        age = (self.get_clock().now() - self.last_perception_stamp).nanoseconds * 1e-9
        if age > self.PERCEPTION_TIMEOUT_S:
            self._publish_zero()
            self.get_logger().warn('No perception data — car stopped.',
                                   throttle_duration_sec=1.0)
            return

        steering_angle = self.th_e + math.atan2(self.k * self.e,
                                                 self.v_actual + self.k_soft)
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        self.smooth_steer = (self.alpha * steering_angle
                             + (1.0 - self.alpha) * self.smooth_steer)

        yaw_rate = self.v_actual * math.tan(self.smooth_steer) / self.wheelbase

        cmd = Twist()
        cmd.linear.x  = self.target_speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e={self.e:+.3f}m  theta_e={math.degrees(self.th_e):+.1f}°  '
            f'delta={math.degrees(self.smooth_steer):+.1f}°  omega={yaw_rate:+.3f}rad/s',
            throttle_duration_sec=0.5)

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def stop(self):
        try:
            for _ in range(5):
                self._publish_zero()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
