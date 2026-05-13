import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
import math


class VehicleKinematicsOdom(Node):

    def __init__(self):
        super().__init__('vehicle_kinematics_odom')

        # ----------------------------
        # SUB / PUB
        # ----------------------------
        self.sub = self.create_subscription(
            AckermannDriveStamped,
            '/drive',
            self.drive_callback,
            10
        )

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.br = tf2_ros.TransformBroadcaster(self)

        # ----------------------------
        # VEHICLE PARAMS
        # ----------------------------
        self.L = 0.20  # wheelbase (20 cm)

        # ----------------------------
        # STATE
        # ----------------------------
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.v = 0.0
        self.delta = 0.0

        # ----------------------------
        # TIME (IMPORTANT FIX)
        # ----------------------------
        self.last_time = self.get_clock().now()

        self.timer = self.create_timer(0.02, self.update)  # 50 Hz

    # ----------------------------
    # DRIVE CALLBACK
    # ----------------------------
    def drive_callback(self, msg):
        self.v = msg.drive.speed
        self.delta = msg.drive.steering_angle

        # safety clamp
        self.delta = max(min(self.delta, 0.4), -0.4)

        # deadzone to prevent drift
        if abs(self.delta) < 0.05:
            self.delta = 0.0

    # ----------------------------
    # UPDATE LOOP
    # ----------------------------
    def update(self):

        # --- REAL dt (FIXED) ---
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time

        # safety clamp (prevents huge jumps)
        dt = min(dt, 0.1)

        # ----------------------------
        # ACKERMANN KINEMATICS
        # ----------------------------
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += (self.v / self.L) * math.tan(self.delta) * dt

        # ----------------------------
        # QUATERNION
        # ----------------------------
        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        now = current_time.to_msg()

        # ----------------------------
        # TF (odom → base_link)
        # ----------------------------
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.br.sendTransform(t)

        # ----------------------------
        # ODOM MESSAGE
        # ----------------------------
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        self.odom_pub.publish(odom)


def main():
    rclpy.init()
    node = VehicleKinematicsOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
