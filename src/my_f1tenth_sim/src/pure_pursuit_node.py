#!/usr/bin/env python3
"""
Pure Pursuit Lane-Keeping Controller
=====================================
Standard geometric pure pursuit using a virtual lookahead point.

Subscribes:
  /perception/crosstrack_error  (std_msgs/Float32)  lateral offset from lane centre (m)
  /odom                         (nav_msgs/Odometry)  vehicle speed

Publishes:
  /cmd_vel  (geometry_msgs/Twist)

Pure Pursuit law:
  alpha = atan2(e, ld)
  delta = atan(2 * L * sin(alpha) / ld)
  omega = v * tan(delta) / L
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult


class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        self.create_subscription(Float32, '/perception/crosstrack_error',
                                 self.crosstrack_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Bool, '/rrt/active', self._rrt_active_cb,
                                 QoSProfile(depth=1,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Bool, '/rrt/goal_reached', self._goal_reached_cb,
                                 QoSProfile(depth=1,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Bool, '/astar/goal_reached', self._goal_reached_cb,
                                 QoSProfile(depth=1,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.e            = 0.0
        self.v_actual     = 0.1
        self.smooth_steer = 0.0
        self._rrt_active  = False    # True while RRT planner is driving
        self._goal_reached = False   # True once RRT planner signals goal reached

        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S  = 1.0
        self._timed_out = False      # True once we've sent the single stop command

        self.target_speed   = 0.5    # m/s
        self.lookahead_dist = 0.5    # m  — virtual target distance ahead
        self.alpha          = 0.4    # EMA smoothing weight
        self.max_steer      = 0.5    # rad — physical joint limit
        self.wheelbase      = 0.28   # m

        self.declare_parameter('target_speed',   self.target_speed)
        self.declare_parameter('lookahead_dist', self.lookahead_dist)
        self.declare_parameter('alpha',          self.alpha)
        self.add_on_set_parameters_callback(self._on_param_change)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Pure Pursuit Controller started.')

    def _on_param_change(self, params):
        for p in params:
            if   p.name == 'target_speed':   self.target_speed   = p.value
            elif p.name == 'lookahead_dist': self.lookahead_dist = p.value
            elif p.name == 'alpha':          self.alpha          = p.value
        return SetParametersResult(successful=True)

    def _rrt_active_cb(self, msg: Bool):
        self._rrt_active = msg.data

    def _goal_reached_cb(self, msg: Bool):
        self._goal_reached = msg.data
        if msg.data:
            # Immediately zero out commands and reset smoothing state
            self._publish_zero()
            self.smooth_steer = 0.0
            self.get_logger().info('Goal reached — controller stopped.')

    def crosstrack_cb(self, msg: Float32):
        self.e = msg.data
        self.last_perception_stamp = self.get_clock().now()
        self._timed_out = False

    def odom_cb(self, msg: Odometry):
        self.v_actual = max(0.05, abs(msg.twist.twist.linear.x))

    def control_loop(self):
        # Hard stop when the RRT planner has declared goal reached
        if self._goal_reached:
            self._publish_zero()
            return

        age = (self.get_clock().now() - self.last_perception_stamp).nanoseconds * 1e-9
        if age > self.PERCEPTION_TIMEOUT_S:
            if not self._timed_out:
                self._publish_zero()
                self._timed_out = True
                self.get_logger().warn('No perception data — car stopped.')
            return

        alpha         = math.atan2(self.e, self.lookahead_dist)
        steering_angle = math.atan(2.0 * self.wheelbase * math.sin(alpha)
                                   / self.lookahead_dist)
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        self.smooth_steer = (self.alpha * steering_angle
                             + (1.0 - self.alpha) * self.smooth_steer)

        yaw_rate = self.v_actual * math.tan(self.smooth_steer) / self.wheelbase

        cmd = Twist()
        cmd.linear.x  = self.target_speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e={self.e:+.3f}m  alpha={math.degrees(alpha):+.1f}°  '
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
    node = PurePursuitController()
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
