#!/usr/bin/env python3
"""
Pure Pursuit Lane-Keeping Controller
=====================================
Subscribes to:
  /perception/crosstrack_error  (std_msgs/Float32)  lateral offset from lane centre (m)
  /perception/heading_error     (std_msgs/Float32)  heading difference (rad)
  /odom                         (nav_msgs/Odometry)  vehicle speed

Publishes:
  /cmd_vel  (geometry_msgs/Twist)  linear.x = wheel angular vel cmd, angular.z = yaw-rate

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


class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Float32, '/perception/crosstrack_error',
                                 self.crosstrack_cb, 10)
        self.create_subscription(Float32, '/perception/heading_error',
                                 self.heading_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.e = 0.0             # crosstrack error (m)
        self.th_e = 0.0          # heading error (rad) — kept but dampened
        self.v_actual = 0.1      # measured forward speed (m/s)
        self.smooth_steer = 0.0  # low-pass filtered steering angle

        # Staleness tracking — stop if no perception update arrives
        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S = 1.0   # seconds before stopping

        # ── Tuning Parameters ─────────────────────────────────────────────────
        # gazebo_ros_ackermann_drive: linear.x = target LINEAR speed (m/s)
        # Start slow — increase once lane-keeping is stable on the full loop
        self.target_speed = 0.5   # m/s — safe starting speed

        # Pure Pursuit lookahead distance (m).
        # δ = atan(2·e·wheelbase / L²)  — shorter L → stronger steering response
        # Too large (0.8m) caps δ at ~13° even at max error; 0.3m triples authority.
        self.L = 0.25             # m — gain = 2*0.28/0.25² = 8.96 rad/m (1cm→5° steer)

        self.max_steer   = 0.5   # max steering angle (rad) — matches URDF joint limit
        self.alpha       = 0.5   # EMA weight — second filter layer on top of detector EMA

        # Heading gain: KEEP LOW or 0. The heading_error from the detector is
        # noisy (jumps to ±65°) and easily destabilises the car.
        # Start at 0.0 (pure crosstrack), increase slowly only if needed on curves.
        self.heading_gain = 0.0

        # ── Vehicle Geometry (from car.xacro) ─────────────────────────────────
        self.wheelbase = 0.28   # metres

        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Pure Pursuit Controller started.')

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
            self.v_actual = 0.05

    # ── Control Loop ──────────────────────────────────────────────────────────
    def control_loop(self):
        # Safety: stop if perception has gone stale
        age = (self.get_clock().now() - self.last_perception_stamp).nanoseconds * 1e-9
        if age > self.PERCEPTION_TIMEOUT_S:
            self._publish_zero()
            self.get_logger().warn('No perception data — car stopped.', throttle_duration_sec=1.0)
            return

        # 1. Pure Pursuit curvature from lateral error
        #    κ = 2·e / L²  →  δ_pp = atan(κ · wheelbase)
        curvature = 2.0 * self.e / (self.L ** 2)
        delta_pp  = math.atan(curvature * self.wheelbase)

        # 2. Blend (clamped) heading error
        steering_angle = delta_pp + self.heading_gain * self.th_e

        # 3. Clamp to physical steering limit
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))

        # 4. Exponential smoothing
        self.smooth_steer = (self.alpha * steering_angle
                             + (1.0 - self.alpha) * self.smooth_steer)

        # 5. Convert steering angle → yaw rate
        #    ω = v_actual · tan(δ) / wheelbase
        yaw_rate = self.v_actual * math.tan(self.smooth_steer) / self.wheelbase

        # 6. Publish
        cmd = Twist()
        cmd.linear.x  = self.target_speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'e={self.e:+.3f}m  δ={math.degrees(self.smooth_steer):+.1f}°  '
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
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        if rclpy.ok():          # only shut down if context is still valid
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
