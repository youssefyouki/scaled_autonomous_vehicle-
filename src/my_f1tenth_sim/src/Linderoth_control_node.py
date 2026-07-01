#!/usr/bin/env python3
"""
Linderoth (2008) Lane-Keeping Controller
==========================================
Lateral law from Linderoth 2008, eq. 4.53 (curvilinear representation).

  e_n   = lateral crosstrack error from lane centre (m)
  e_th  = heading error relative to road tangent    (rad)
  k1    = centre-return gain   (start: 1.0)
  k2    = crosstrack gain      (start: 1.0)

RAW FORMULA (from thesis):
  num =  -cos(e_th)*e_n  -  (k1+k2)*sin(e_th)
  den =   k1  -  (k1+k2)*cos(e_th)  +  sin(e_th)*e_n
  δ   =  atan(num / den)

EQUILIBRIUM CHECK (e_n=0, e_th=0):
  num = 0
  den = k1 - (k1+k2) = -k2   ← ALWAYS NEGATIVE at rest

  atan(0 / -k2) = 0  ✓  (atan of zero is always zero — accidentally correct)
  atan2(0, -k2) = π  ✗  (atan2 with negative den reads the wrong quadrant)

  The original atan works at equilibrium but fails on curves when den flips
  sign mid-range (wrong quadrant → wrong-direction steering correction).

CORRECT FIX — negate both num and den before atan2:
  atan2(-num, -den) = atan2(0, +k2) = 0  ✓  at equilibrium
  And atan2 now handles any quadrant correctly on curves.

Equivalent rewrite:
  num2 =  cos(e_th)*e_n  +  (k1+k2)*sin(e_th)
  den2 =  (k1+k2)*cos(e_th)  -  sin(e_th)*e_n  -  k1
  δ    =  atan2(num2, den2)

  den2 at equilibrium = (k1+k2) - k1 = k2 > 0  ✓

Tuning sequence:
  1. heading_scale=0, k1=1.0, k2=1.0 → verify crosstrack correction on a straight
  2. Adjust k2 (tighter centering) / k1 (damp oscillation) until straight is stable
  3. Raise heading_scale toward 1.0 for curve anticipation
  4. If heading correction acts backwards, set heading_scale negative (e.g. -0.5)
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32
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
        self.create_subscription(Bool, '/rrt/active', self._rrt_active_cb,
                                 QoSProfile(depth=1,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.e_n          = 0.0
        self.e_th         = 0.0
        self.v_actual     = 0.05
        self.smooth_steer = 0.0
        self._first_tick  = True
        self._rrt_active  = False    # True while RRT planner is driving

        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S  = 1.0
        self._timed_out = False      # True once we've sent the single stop command

        # ── Tunable parameters ─────────────────────────────────────────────
        self.target_speed   = 0.50   # m/s
        self.k1             = 1.0    # centre-return gain
        self.k2             = 1.0    # crosstrack gain
        # heading_scale:  0 = pure crosstrack only (start here)
        #                 1 = full anticipatory heading
        #                -1 = if heading is acting backwards, try negative
        self.heading_scale  = 0.50
        self.alpha          = 0.4    # EMA smoothing (0=frozen, 1=no filter)
        self.max_steer      = 0.5    # rad — physical joint limit
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

    def _rrt_active_cb(self, msg: Bool):
        self._rrt_active = msg.data

    def crosstrack_cb(self, msg: Float32):
        self.e_n = msg.data
        self.last_perception_stamp = self.get_clock().now()
        self._timed_out = False

    def heading_cb(self, msg: Float32):
        self.e_th = max(-math.radians(20.0), min(math.radians(20.0), msg.data))

    def odom_cb(self, msg: Odometry):
        self.v_actual = max(0.05, abs(msg.twist.twist.linear.x))

    def control_loop(self):
        age = (self.get_clock().now() - self.last_perception_stamp).nanoseconds * 1e-9
        if age > self.PERCEPTION_TIMEOUT_S:
            if not self._timed_out:
                self._publish_zero()
                self._timed_out = True
                self.get_logger().warn('No perception data — car stopped.')
            return

        e_n  = self.e_n
        e_th = self.heading_scale * self.e_th
        k12  = self.k1 + self.k2

        # ── Linderoth eq. 4.53, negated num+den so atan2 works correctly ──
        # At equilibrium (e_n=0, e_th=0):
        #   num2 = 0,  den2 = k2 > 0  →  atan2(0, k2) = 0  ✓
        num2 =  math.cos(e_th) * e_n  +  k12 * math.sin(e_th)
        den2 =  k12 * math.cos(e_th)  -  math.sin(e_th) * e_n  -  self.k1

        steering_angle = math.atan2(num2, den2)
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        if self._first_tick:
            self.smooth_steer = steering_angle
            self._first_tick  = False
        else:
            self.smooth_steer = (self.alpha * steering_angle
                             + (1.0 - self.alpha) * self.smooth_steer)

        yaw_rate = self.target_speed * math.tan(self.smooth_steer) / self.wheelbase

        cmd = Twist()
        cmd.linear.x  = self.target_speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e_n={e_n:+.3f}m  e_th_raw={math.degrees(self.e_th):+.1f}°  '
            f'e_th_in={math.degrees(e_th):+.1f}°  '
            f'num2={num2:+.4f}  den2={den2:+.4f}  '
            f'delta={math.degrees(self.smooth_steer):+.1f}°  omega={yaw_rate:+.3f}',
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