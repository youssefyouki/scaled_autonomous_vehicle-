#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import socket
import cv2
import numpy as np
import pickle
import math

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TransformStamped
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import tf2_ros


HOST = "0.0.0.0"
PORT = 5011


class VectorCommNode(Node):

    def __init__(self):
        super().__init__("vector_comm_node")

        # ---------- ROS ----------
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_img = self.create_publisher(Image, "/camera/image_raw", 10)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", 10)

        self.bridge = CvBridge()
        self.br = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(
            AckermannDriveStamped,
            "drive",
            self.drive_cb,
            10
        )

        # ---------- SOCKET ----------
        self.server = socket.socket()
        self.server.bind((HOST, PORT))
        self.server.listen(1)

        self.get_logger().info("Waiting for connection...")
        self.conn, _ = self.server.accept()
        self.get_logger().info("Connected!")

        self.conn.setblocking(False)  # 🔥 KEY FIX

        # ---------- STATE ----------
        self.vel = 0.0
        self.steer = 0.0

        self.pos = [0.0, 0.0, 0.0]
        self.last_time = self.get_clock().now()

        self.latest_scan = None

        # ---------- TIMER (MAIN LOOP) ----------
        self.timer = self.create_timer(0.03, self.loop)  # ~33 Hz

    # ================= DRIVE =================
    def drive_cb(self, msg):
        self.vel = msg.drive.speed
        self.steer = msg.drive.steering_angle

    # ================= SAFE RECV =================
    def try_recv(self, size):
        try:
            data = self.conn.recv(size)
            return data if len(data) > 0 else None
        except:
            return None

    # ================= FRAME =================
    def get_frame(self):
        try:
            self.conn.sendall(b'F')

            size = self.conn.recv(8)
            if len(size) < 8:
                return None

            img_size = int.from_bytes(size, "big")

            data = b''
            while len(data) < img_size:
                packet = self.conn.recv(img_size - len(data))
                if not packet:
                    return None
                data += packet

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return cv2.resize(frame, (640, 480))

        except:
            return None

    # ================= META =================
    def get_meta(self):
        try:
            self.conn.sendall(b'M')

            size = self.conn.recv(4)
            if len(size) < 4:
                return None, None

            meta_size = int.from_bytes(size, "big")

            data = b''
            while len(data) < meta_size:
                packet = self.conn.recv(meta_size - len(data))
                if not packet:
                    return None, None
                data += packet

            s, y = data.decode().split(',')
            return float(s) / 10, float(y)

        except:
            return None, None

    # ================= LIDAR (NON BLOCKING CACHE) =================
    def update_lidar(self):
        try:
            self.conn.sendall(b'L')

            size = self.conn.recv(8)
            if len(size) < 8:
                return

            lidar_size = int.from_bytes(size, "big")

            data = b''
            while len(data) < lidar_size:
                packet = self.conn.recv(lidar_size - len(data))
                if not packet:
                    return
                data += packet

            self.latest_scan = pickle.loads(data)

        except:
            pass  # never block

    # ================= PUBLISH =================
    def publish_scan(self, scan, t):
        msg = LaserScan()
        msg.header.stamp = t.to_msg()
        msg.header.frame_id = "laser"

        angle_min = -math.pi
        angle_max = math.pi
        n = 360

        inc = (angle_max - angle_min) / n
        ranges = [float("inf")] * n

        angles = np.radians(scan["angles"])
        dists = np.array(scan["ranges"]) / 1000.0

        for a, d in zip(angles, dists):
            i = int((a - angle_min) / inc)
            if 0 <= i < n:
                ranges[i] = d

        msg.angle_min = angle_min
        msg.angle_max = angle_max
        msg.angle_increment = inc
        msg.ranges = ranges

        self.pub_scan.publish(msg)

    # ================= MAIN LOOP =================
    def loop(self):

        t = self.get_clock().now()

        # ---------- FRAME ----------
        frame = self.get_frame()
        if frame is not None:
            self.pub_img.publish(
                self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            )
            cv2.imshow("cam", frame)

        # ---------- META ----------
        speed, yaw = self.get_meta()
        if speed is not None:

            dt = (t - self.last_time).nanoseconds * 1e-9
            dt = min(dt, 0.1)

            theta = math.radians(yaw)

            self.pos[0] += speed * math.cos(theta) * dt
            self.pos[1] += speed * math.sin(theta) * dt

            qz = math.sin(theta / 2)
            qw = math.cos(theta / 2)

            tf = TransformStamped()
            tf.header.stamp = t.to_msg()
            tf.header.frame_id = "odom"
            tf.child_frame_id = "base_link"

            tf.transform.translation.x = self.pos[0]
            tf.transform.translation.y = self.pos[1]
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw

            self.br.sendTransform(tf)

            odom = Odometry()
            odom.header = tf.header
            odom.pose.pose.position.x = self.pos[0]
            odom.pose.pose.position.y = self.pos[1]
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw

            self.pub_odom.publish(odom)

            self.last_time = t

        # ---------- LIDAR ----------
        self.update_lidar()
        if self.latest_scan is not None:
            self.publish_scan(self.latest_scan, t)

        # ---------- CONTROL ----------
        pwm = int((abs(self.vel) / 0.5) * 255)
        steer = int(((self.steer / 0.5) * 25) + 96)

        try:
            self.conn.sendall(f"{pwm},{steer}\n".encode())
        except:
            pass

        if cv2.waitKey(1) == ord('q'):
            rclpy.shutdown()


# ================= MAIN =================
def main():
    rclpy.init()

    node = VectorCommNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
