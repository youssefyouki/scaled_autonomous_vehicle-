#!/usr/bin/env python3
"""
Linderoth (2008) Lane-Keeping Controller
==========================================
Lateral law from Linderoth 2008, eq. 4.53, with an explicit heading_scale
parameter that controls how much of the (anticipatory) heading error enters
the formula.

  heading_scale = 0.0  →  pure crosstrack:  δ = atan(e_n / k2)
  heading_scale = 1.0  →  full Linderoth formula with look-ahead heading

The look-ahead heading from the lane detector reflects upcoming bends before
crosstrack error builds; this causes early turning that cannot be tuned away
with k1/k2 alone.  heading_scale lets you dial that anticipation from zero up.

Lateral law:
  e_th_in     = heading_scale * heading_error   (heading_error negated for convention)
  numerator   = -cos(e_th_in)*e_n - (k1+k2)*sin(e_th_in)
  denominator =  k1 - (k1+k2)*cos(e_th_in) + sin(e_th_in)*e_n
  delta       = atan(numerator / denominator)
  omega       = v * tan(delta) / L
"""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult


class LinderothController(Node):
    def __init__(self):
        super().__init__('linderoth_controller')

        self.create_subscription(Float32, '/perception/crosstrack_error',
                                 self.crosstrack_cb, 10)
        self.create_subscription(Float32, '/perception/heading_error',
                                 self.heading_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.e_n          = 0.0
        self.e_th         = 0.0
        self.v_actual     = 0.05
        self.smooth_steer = 0.0

        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S  = 1.0

        self.target_speed   = 0.35   # m/s
        self.k1             = 0.3    # lateral stability gain  (must be < k2)
        self.k2             = 0.6    # crosstrack sensitivity
        self.heading_scale  = 0.0    # 0 = pure crosstrack, 1 = full anticipatory heading
        self.alpha          = 0.4    # EMA smoothing weight
        self.max_steer      = 0.5    # rad
        self.wheelbase      = 0.28   # m

        self.declare_parameter('target_speed',  self.target_speed)
        self.declare_parameter('k1',            self.k1)
        self.declare_parameter('k2',            self.k2)
        self.declare_parameter('heading_scale', self.heading_scale)
        self.declare_parameter('alpha',         self.alpha)
        self.add_on_set_parameters_callback(self._on_param_change)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Linderoth Controller started.')

    def _on_param_change(self, params):
        for p in params:
            if   p.name == 'target_speed':  self.target_speed  = p.value
            elif p.name == 'k1':            self.k1            = p.value
            elif p.name == 'k2':            self.k2            = p.value
            elif p.name == 'heading_scale': self.heading_scale = p.value
            elif p.name == 'alpha':         self.alpha         = p.value
        return SetParametersResult(successful=True)

    def crosstrack_cb(self, msg: Float32):
        self.e_n = msg.data
        self.last_perception_stamp = self.get_clock().now()

    def heading_cb(self, msg: Float32):
        self.e_th = max(-math.radians(20.0), min(math.radians(20.0), msg.data))

    def odom_cb(self, msg: Odometry):
        self.v_actual = max(0.05, abs(msg.twist.twist.linear.x))

    def control_loop(self):
        age = (self.get_clock().now() - self.last_perception_stamp).nanoseconds * 1e-9
        if age > self.PERCEPTION_TIMEOUT_S:
            self._publish_zero()
            self.get_logger().warn('No perception data — car stopped.',
                                   throttle_duration_sec=1.0)
            return

        e_n  = self.e_n
        # heading_scale=0 → e_th=0 → formula reduces to δ=atan(e_n/k2)
        # heading_scale=1 → full anticipatory look-ahead heading
        # The negation aligns the look-ahead heading convention with Linderoth eq. 4.53.
        e_th = self.heading_scale * (-self.e_th)
        k12  = self.k1 + self.k2

        numerator   = -math.cos(e_th) * e_n  -  k12 * math.sin(e_th)
        denominator =  self.k1  -  k12 * math.cos(e_th)  +  math.sin(e_th) * e_n

        if abs(denominator) < 1e-6:
            steering_angle = math.copysign(math.pi / 2.0, numerator)
        else:
            steering_angle = math.atan(numerator / denominator)

        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        self.smooth_steer = (self.alpha * steering_angle
                             + (1.0 - self.alpha) * self.smooth_steer)

        yaw_rate = self.v_actual * math.tan(self.smooth_steer) / self.wheelbase

        cmd = Twist()
        cmd.linear.x  = self.target_speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e_n={e_n:+.3f}m  e_th={math.degrees(e_th):+.1f}°  '
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
    node = LinderothController()
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
