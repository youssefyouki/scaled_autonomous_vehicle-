#!/usr/bin/env python3
"""
Linderoth (2008) Lane-Keeping Controller
==========================================
Lateral law  — Linderoth 2008, eq. 4.53 (curvilinear representation):

    δ = atan( ( -cos(e_th)·e_n  -  (k1+k2)·sin(e_th) )
              / ( k1  -  (k1+k2)·cos(e_th)  +  sin(e_th)·e_n ) )

    e_n   = lateral distance error from lane centre  (crosstrack, m)
    e_th  = heading error relative to road profile   (rad)
    k1,k2 = stability gains (dimensionless)

Longitudinal law — discrete PI controller (eq. 4.52):

    e_v          = V_desired - V_actual
    A_car_long   = Kp · ( e_v  +  Ti · Σ(Δt · e_n) )      Ti = Ki/Kp
    V_desired_k+1 = V_desired_k  +  Δt · A_car_long
    (result clamped to [0, v_max])

Subscribes to:
  /perception/crosstrack_error  (std_msgs/Float32)  lateral offset (m)
  /perception/heading_error     (std_msgs/Float32)  heading difference (rad)
  /odom                         (nav_msgs/Odometry)  vehicle speed

Publishes:
  /cmd_vel  (geometry_msgs/Twist)
      linear.x  = commanded speed  (m/s)   — updated by PI
      angular.z = yaw rate         (rad/s) — derived from δ
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

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Float32, '/perception/crosstrack_error',
                                 self.crosstrack_cb, 10)
        self.create_subscription(Float32, '/perception/heading_error',
                                 self.heading_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Perception state ──────────────────────────────────────────────────
        self.e_n  = 0.0    # crosstrack / lateral distance error (m)
        self.e_th = 0.0    # heading error (rad)

        # Staleness tracking — stop if no perception update arrives
        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S  = 1.0

        # ── Odometry state ────────────────────────────────────────────────────
        self.v_actual = 0.05   # measured forward speed (m/s); floor avoids /0

        # ── Lateral tuning (Linderoth gains) ──────────────────────────────────
        # Near zero error the law linearises to  δ ≈ e_n/k2 + (k1/k2+1)·e_th
        # so k2 is the crosstrack-to-steer sensitivity.  k2 must be > k1 for
        # damped straight-line behaviour; too-small k2 causes oscillation.
        self.k1 = 0.3
        self.k2 = 0.6

        # Heading-error clamp (suppress large detector noise spikes)
        self.MAX_HEADING_ERR = math.radians(20.0)

        # Steering output clamp — matches URDF joint limit (rad)
        self.max_steer = 0.5

        # EMA smoothing weight for steering (0=frozen, 1=no filter)
        self.alpha = 0.4
        self.smooth_steer = 0.0

        # Scale the heading-error contribution to the formula.
        # The detector evaluates e_th ~0.28 m ahead (look-ahead), which makes the
        # heading term fire early and produce large angles in curves.  Values < 1
        # damp both the early onset and the curve amplitude without affecting the
        # crosstrack correction.  Start at 0.5; raise toward 1.0 if the car is
        # slow to follow curves, lower toward 0.3 if it still turns too fast.
        self.heading_scale = 0.4

        # ── Longitudinal tuning (PI) ───────────────────────────────────────────
        # v_ref  — fixed target speed; never changes.
        # v_cmd  — PI-integrated speed command sent to the car; starts at 0
        #          and ramps up toward v_ref.  Keeping these separate prevents
        #          the runaway where v_cmd grows every loop because e_v > 0.
        self.v_ref = 0.35      # target speed (m/s)
        self.v_cmd = 0.0       # current speed command (PI output)

        self.Kp = 0.8
        self.Ki = 0.1
        self.v_max = 0.45      # absolute speed ceiling (m/s)
        self.v_min = 0.0

        # Integrator state and anti-windup clamp
        self.integral_ev    = 0.0
        self.integral_limit = 1.0

        # ── Vehicle geometry (from car.xacro) ─────────────────────────────────
        self.wheelbase = 0.28   # metres

        # ── ROS2 parameter declarations (enables live tuning via controller panel)
        self.declare_parameter('k1',            self.k1)
        self.declare_parameter('k2',            self.k2)
        self.declare_parameter('v_ref',         self.v_ref)
        self.declare_parameter('heading_scale', self.heading_scale)
        self.declare_parameter('alpha',         self.alpha)
        self.declare_parameter('max_steer',     self.max_steer)
        self.declare_parameter('Kp',            self.Kp)
        self.declare_parameter('Ki',            self.Ki)
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Control loop at 20 Hz ─────────────────────────────────────────────
        self.DT    = 0.05
        self.timer = self.create_timer(self.DT, self.control_loop)
        self.get_logger().info('Linderoth Controller started.')

    # ── Parameter callback ────────────────────────────────────────────────────

    def _on_param_change(self, params):
        for p in params:
            if   p.name == 'k1':            self.k1 = p.value
            elif p.name == 'k2':            self.k2 = p.value
            elif p.name == 'v_ref':         self.v_ref = p.value
            elif p.name == 'heading_scale': self.heading_scale = p.value
            elif p.name == 'alpha':         self.alpha = p.value
            elif p.name == 'max_steer':     self.max_steer = p.value
            elif p.name == 'Kp':            self.Kp = p.value
            elif p.name == 'Ki':            self.Ki = p.value
        return SetParametersResult(successful=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def crosstrack_cb(self, msg: Float32):
        self.e_n = msg.data
        self.last_perception_stamp = self.get_clock().now()

    def heading_cb(self, msg: Float32):
        self.e_th = max(-self.MAX_HEADING_ERR,
                        min(self.MAX_HEADING_ERR, msg.data))

    def odom_cb(self, msg: Odometry):
        self.v_actual = max(0.05, abs(msg.twist.twist.linear.x))

    # ── Control Loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        # ── Safety: stop on stale perception ──────────────────────────────────
        age = (self.get_clock().now() - self.last_perception_stamp).nanoseconds * 1e-9
        if age > self.PERCEPTION_TIMEOUT_S:
            self._publish_zero()
            self.get_logger().warn('No perception data — car stopped.',
                                   throttle_duration_sec=1.0)
            return

        # ── 1. Linderoth lateral control law (eq. 4.53) ───────────────────────
        #
        #   numerator   = -cos(e_th)·e_n  -  (k1+k2)·sin(e_th)
        #   denominator =  k1  -  (k1+k2)·cos(e_th)  +  sin(e_th)·e_n
        #
        # The denominator can theoretically pass through zero for very large
        # errors; the math.atan2(num, den) form handles the ±π/2 case safely.
        e_n  = self.e_n
        # The detector publishes e_th = -atan(lane_slope), so positive e_th means
        # the lane curves LEFT (road bending left, car heading right relative to road).
        # Linderoth eq. 4.53 expects the OPPOSITE convention: positive e_th = car
        # heading RIGHT.  Negate here so the formula corrects in the right direction.
        # heading_scale < 1 reduces the look-ahead anticipation and curve amplitude.
        e_th = -self.e_th * self.heading_scale
        k12  = self.k1 + self.k2

        numerator   = -math.cos(e_th) * e_n  -  k12 * math.sin(e_th)
        denominator =  self.k1  -  k12 * math.cos(e_th)  +  math.sin(e_th) * e_n

        # Paper uses plain atan (result in (-π/2, π/2)).  atan2 is WRONG here:
        # near equilibrium the denominator is always negative (-k2), so atan2
        # returns ≈ ±π (maximum steering) for any small error.
        if abs(denominator) < 1e-6:
            steering_angle = math.copysign(math.pi / 2, numerator)
        else:
            steering_angle = math.atan(numerator / denominator)

        # Clamp to physical steering limit
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        # EMA smoothing
        self.smooth_steer = (self.alpha * steering_angle
                             + (1.0 - self.alpha) * self.smooth_steer)

        # Convert steering angle → yaw rate:  ω = v·tan(δ) / L
        yaw_rate = self.v_actual * math.tan(self.smooth_steer) / self.wheelbase

        # ── 2. PI longitudinal controller (eq. 4.52) ──────────────────────────
        # e_v uses the fixed reference — v_ref never changes so the error
        # correctly drives v_cmd toward the target without runaway.
        e_v = self.v_ref - self.v_actual

        self.integral_ev = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral_ev + self.DT * e_v)
        )

        Ti     = self.Ki / self.Kp
        a_long = self.Kp * (e_v + Ti * self.integral_ev)

        self.v_cmd = max(self.v_min,
                         min(self.v_max, self.v_cmd + self.DT * a_long))

        # ── 3. Publish ─────────────────────────────────────────────────────────
        cmd = Twist()
        cmd.linear.x  = self.v_cmd
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e_n={e_n:+.3f}m  e_th={math.degrees(e_th):+.1f}°  '
            f'δ={math.degrees(self.smooth_steer):+.1f}°  '
            f'ω={yaw_rate:+.3f} rad/s  '
            f'v_cmd={self.v_cmd:.2f}m/s  v_act={self.v_actual:.2f}m/s',
            throttle_duration_sec=0.5
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def stop(self):
        """Publish zero velocity — called on shutdown."""
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