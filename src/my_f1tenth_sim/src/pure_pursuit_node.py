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
        self.e        = 0.0    # crosstrack error (m)
        self.prev_e   = 0.0    # previous crosstrack error for derivative term
        self.th_e     = 0.0    # heading error (rad)
        self.v_actual = 0.1    # measured forward speed (m/s)
        self.smooth_steer = 0.0

        # Staleness tracking — stop if no perception update arrives
        self.last_perception_stamp = self.get_clock().now()
        self.PERCEPTION_TIMEOUT_S = 1.0

        # ── Tuning Parameters (matching Stanley controller) ────────────────────
        self.target_speed = 0.5   # m/s

        # Stanley law: δ = heading_gain·θ_e + atan(k·e_pred / (v + k_soft))
        # k_d: derivative look-ahead — predicts where the error will be when
        # the car reaches the nearest camera point (≈0.15 s at 0.5 m/s).
        self.k         = 1.0
        self.k_soft    = 0.4
        self.k_d       = 0.15
        self.heading_gain = 0.0

        self.max_steer = 0.5
        self.alpha     = 0.5

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

        # 1. Stanley law with derivative crosstrack prediction
        DT              = 0.05
        e_rate          = (self.e - self.prev_e) / DT
        self.prev_e     = self.e
        e_pred          = self.e + self.k_d * e_rate
        atan_term       = math.atan2(self.k * e_pred, self.v_actual + self.k_soft)
        steering_angle  = self.heading_gain * self.th_e + atan_term

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
            f'e={self.e:+.3f}m  e_pred={e_pred:+.3f}m  '
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
