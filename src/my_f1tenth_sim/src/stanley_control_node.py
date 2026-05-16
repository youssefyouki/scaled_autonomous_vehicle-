#!/usr/bin/env python3
"""
Stanley Lane-Keeping Controller
================================
Subscribes to:
  /perception/crosstrack_error  (std_msgs/Float32)  lateral offset from lane centre (m)
  /perception/heading_error     (std_msgs/Float32)  heading difference (rad)
  /odom                         (nav_msgs/Odometry)  vehicle speed

Publishes:
  /cmd_vel  (geometry_msgs/Twist)  linear.x = target LINEAR speed (m/s),
                                    angular.z = yaw rate (rad/s)

Stanley steering law:
  δ = θ_e + arctan(k · e / (v + k_soft))

NOTE: gazebo_ros_ackermann_drive interprets linear.x as the target LINEAR speed (m/s)
      and angular.z as desired yaw rate (rad/s).
      Keep target_speed low (≤1.0 m/s) until the track loop is stable.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math


class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Float32, '/perception/crosstrack_error',
                                 self.crosstrack_cb, 10)
        self.create_subscription(Float32, '/perception/heading_error',
                                 self.heading_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.e          = 0.0    # crosstrack error (m)
        self.th_e       = 0.0   # heading error (rad)
        self.v_actual   = 0.1   # measured forward speed (m/s)
        self.smooth_steer = 0.0 # EMA-filtered steering angle (rad)

        # Staleness tracking — stop if no perception update arrives
        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S  = 1.0   # seconds before stopping

        # ── Tuning Parameters ─────────────────────────────────────────────────
        self.target_speed = 0.5   # m/s — safe starting speed (matches pure pursuit)

        # Stanley gains
        # k      : crosstrack gain — higher = more aggressive lateral correction
        # k_soft : softening constant — prevents division-by-zero and reduces
        #          oscillation at low speeds (increase if car wiggles at <0.3 m/s)
        self.k      = 1.5
        self.k_soft = 0.5

        self.max_steer = 0.5   # max steering angle (rad) — matches URDF joint limit
        self.alpha     = 0.5   # EMA weight — matches pure pursuit (double-filtered
                               # with detector EMA for smooth commands)

        # ── Vehicle Geometry (from car.xacro) ─────────────────────────────────
        self.wheelbase = 0.28   # metres

        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Stanley Controller started.')

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def crosstrack_cb(self, msg: Float32):
        self.e = msg.data
        self.last_perception_stamp = self.get_clock().now()

    def heading_cb(self, msg: Float32):
        # Clamp heading error to ±20° to suppress detector noise spikes
        MAX_TH = math.radians(20.0)
        self.th_e = max(-MAX_TH, min(MAX_TH, msg.data))

    def odom_cb(self, msg: Odometry):
        self.v_actual = abs(msg.twist.twist.linear.x)
        if self.v_actual < 0.05:
            self.v_actual = 0.05   # guard against division by zero in Stanley law

    # ── Control Loop ──────────────────────────────────────────────────────────
    def control_loop(self):
        # Safety: stop if perception has gone stale
        age = (self.get_clock().now() - self.last_perception_stamp).nanoseconds * 1e-9
        if age > self.PERCEPTION_TIMEOUT_S:
            self._publish_zero()
            self.get_logger().warn('No perception data — car stopped.',
                                   throttle_duration_sec=1.0)
            return

        # 1. Stanley steering law
        #    δ = θ_e + arctan(k · e / (v + k_soft))
        atan_term     = math.atan2(self.k * self.e, self.v_actual + self.k_soft)
        steering_angle = self.th_e + atan_term

        # 2. Clamp to physical steering limit
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        # 3. EMA smoothing (same layer as pure pursuit's alpha)
        self.smooth_steer = (self.alpha * steering_angle
                             + (1.0 - self.alpha) * self.smooth_steer)

        # 4. Convert steering angle → yaw rate
        #    ω = v · tan(δ) / wheelbase
        yaw_rate = self.v_actual * math.tan(self.smooth_steer) / self.wheelbase

        # 5. Publish
        cmd = Twist()
        cmd.linear.x  = self.target_speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e={self.e:+.3f}m  θ_e={math.degrees(self.th_e):+.1f}°  '
            f'δ={math.degrees(self.smooth_steer):+.1f}°  '
            f'ω={yaw_rate:+.3f} rad/s  v={self.v_actual:.2f}m/s',
            throttle_duration_sec=0.5
        )

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def stop(self):
        """Publish zero velocity to halt the car — called on shutdown."""
        try:
            for _ in range(5):
                self._publish_zero()
        except Exception:
            pass  # publisher may already be gone; that's fine


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
