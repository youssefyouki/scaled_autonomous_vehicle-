#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO
import socket
import cv2
import math
import pickle
import threading
#from perception_demo import demo_Code

import rclpy
import tf2_ros
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from ackermann_msgs.msg import AckermannDriveStamped

#####################################
# === Configuration ===
HOST = '0.0.0.0'
PORT = 5011
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
#####################################

#################################################
rclpy.init(args=None)
node = rclpy.create_node('Vector_Comm_Node')
br = tf2_ros.TransformBroadcaster(node)
pub_Odom = node.create_publisher(Odometry,'/odom', 10)
bridge = CvBridge()

pub_Image = node.create_publisher(Image, '/camera/image_raw', 10)
pub_Scan = node.create_publisher(LaserScan, '/scan', 10)

Vel = 0
steer = 0

def cmd_ackermann_callback(message):
 global Vel, steer
 Vel = message.drive.speed
 steer = message.drive.steering_angle

sub_Ack = node.create_subscription(AckermannDriveStamped, 'drive', cmd_ackermann_callback, 1)
#################################################

#################################################
# Spin in a separate thread
def spin_thread():
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass  # suppress clean shutdown exception

thread = threading.Thread(target=spin_thread, daemon=True)
thread.start()
#################################################

#####################################
def publish_odom(speed,yaw_rate,last_time,pos):
        global node, br, pub_Odom
        Veh_L = 0.2
        
        # --- REAL dt (FIXED) ---
        current_time = node.get_clock().now()
        dt = (current_time - last_time).nanoseconds * 1e-9

        # safety clamp (prevents huge jumps)
        dt = min(dt, 0.1)

        # ----------------------------
        # ACKERMANN KINEMATICS
        # ----------------------------
        theta = np.radians(yaw_rate)#pos[2] + yaw_rate* dt #(speed / Veh_L) * math.tan(pos[2]) * dt
        x = pos[0] + speed * math.cos(theta) * dt
        y = pos[1] + speed * math.sin(theta) * dt
        
        # ----------------------------
        # QUATERNION
        # ----------------------------
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)

        now = current_time.to_msg()

        # ----------------------------
        # TF (odom → base_link)
        # ----------------------------
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        br.sendTransform(t)

        # ----------------------------
        # ODOM MESSAGE
        # ----------------------------
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        pub_Odom.publish(odom)
        pos = [x,y,theta]
        return current_time,pos
#####################################

#####################################
def publish_image(frame):
    try:
        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        pub_Image.publish(msg)

    except Exception as e:
        print(f"[ROS2 IMAGE ERROR] {e}")
#####################################

#####################################
def publish_scan(lidar_scan):
    try:
        msg = LaserScan()

        now = node.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "laser"

        # === FIXED GRID (SLAM SAFE) ===
        angle_min = -math.pi
        angle_max = math.pi
        num_points = 360  # 1° resolution

        angle_increment = (angle_max - angle_min) / num_points

        ranges = [float('inf')] * num_points

        # incoming data
        angles = np.radians(lidar_scan["angles"])
        distances = np.array(lidar_scan["ranges"]) / 1000.0  # mm → m

        for a, d in zip(angles, distances):
            idx = int((a - angle_min) / angle_increment)
            if 0 <= idx < num_points:
                ranges[idx] = d

        msg.angle_min = angle_min
        msg.angle_max = angle_max
        msg.angle_increment = angle_increment

        msg.range_min = 0.15
        msg.range_max = 12.0

        msg.ranges = ranges

        pub_Scan.publish(msg)

    except Exception as e:
        print(f"[ROS2 LIDAR ERROR] {e}")
#####################################


#####################################
def subscribe_drive():
  global Vel, steer
  print(Vel, steer)
  
  steerang = int(((steer/0.5)*25) + 96)
  pwm = int((np.abs(Vel)/0.5)*255)#90#demo_Code(frame)
  return pwm, steerang
#####################################

#####################################
def pull_frame(conn):
    try:
        # Request a new frame
        conn.sendall(b'F')

        # === Receive image size ===
        size_data = conn.recv(8)
        if len(size_data) < 8:
            print("[SERVER] Incomplete image size.")
            return None

        img_size = int.from_bytes(size_data, byteorder='big')

        # === Receive image data ===
        img_bytes = b''
        while len(img_bytes) < img_size:
            packet = conn.recv(min(4096, img_size - len(img_bytes)))
            if not packet:
                return None
            img_bytes += packet
        
        # === Decode frame ===
        frame_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        return frame

    except Exception as e:
        print(f"[SERVER] Error while pulling frame: {e}")
        return None
#####################################

#####################################
def pull_meta(conn):
    try:
      
        # Request a new frame
        conn.sendall(b'M')

        # === Receive metadata size ===
        meta_size_data = conn.recv(4)
        if len(meta_size_data) < 4:
            print("[SERVER] Incomplete metadata size.")
            return None, None, None

        meta_size = int.from_bytes(meta_size_data, byteorder='big')
        
        # === Receive metadata ===
        meta_bytes = b''
        while len(meta_bytes) < meta_size:
            packet = conn.recv(min(1024, meta_size - len(meta_bytes)))
            if not packet:
                return None, None, None
            meta_bytes += packet

        meta_str = meta_bytes.decode('utf-8')

        # Parse "speed,yawrate"
        try:
            speed_str, yawrate_str = meta_str.split(',')
            speed = float(speed_str)/10
            yaw_rate = float(yawrate_str)
        except Exception:
            print(f"[SERVER] Invalid metadata format: {meta_str}")
            speed, yaw_rate = None, None

        return speed, yaw_rate

    except Exception as e:
        print(f"[SERVER] Error while pulling meta: {e}")
        return None, None
#####################################

#####################################
def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("Socket disconnected")
        data += packet
    return data
    
def pull_lidar(conn):
    lidar_scan = None

    try:
        # send request
        conn.sendall(b'L')

        # ⏱️ short timeout (critical)
        conn.settimeout(0.2)

        # === SIZE ===
        lidar_size_data = recv_exact(conn, 8)
        lidar_size = int.from_bytes(lidar_size_data, 'big')

        # sanity check
        if lidar_size <= 0 or lidar_size > 10_000_000:
            print(f"[SERVER] Invalid LIDAR size: {lidar_size}")
            return None

        # === DATA ===
        lidar_bytes = recv_exact(conn, lidar_size)

        try:
            lidar_scan = pickle.loads(lidar_bytes)
        except Exception as e:
            print(f"[SERVER] LIDAR decode error: {e}")
            lidar_scan = None

    except socket.timeout:
        # ✅ THIS IS NORMAL (lidar not ready yet)
        print("[SERVER] LIDAR timeout (no data yet)")
        lidar_scan = None

    except Exception as e:
        print(f"[SERVER] Error while pulling lidar: {e}")
        lidar_scan = None

    finally:
        conn.settimeout(None)  # restore blocking mode

    return lidar_scan
#####################################

#####################################
def main(args=None):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[SERVER] Listening on {HOST}:{PORT}...")

    conn, addr = server_socket.accept()
    print(f"[SERVER] Connected to {addr}")

    try:
        last_time = node.get_clock().now()
        pos = [0,0,0]
        while True:
            print('===============================')
            # === Receive Frame Data ===
            frame = pull_frame(conn)
                
            if frame is not None:
                cv2.imshow("Client Video Feed", frame)
                publish_image(frame)                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("[SERVER] No frame received or decoding failed.")
                break
            print("Frame is received")
            
            # === Receive Meta Data ===
            speed, yaw_rate = pull_meta(conn)
            if speed is not None and yaw_rate is not None:
                last_time,pos = publish_odom(speed,yaw_rate,last_time,pos)
                print(f"[SENSOR] Speed: {speed:.3f} m/s | YawRate: {yaw_rate:.2f}°")
             
            print("Meta is received")
                 
            # === Receive Lidar Data ===
            lidar_scan = pull_lidar(conn)
            if lidar_scan is not None:
              publish_scan(lidar_scan)
              print("lidar is received")
            
            # === Send control values ===
            pwm, steer = subscribe_drive()
            control_msg = f"{pwm},{steer}\n"
            conn.sendall(control_msg.encode('utf-8'))
            #Send the pwm and steer
            print(pwm, steer)

    except KeyboardInterrupt:
        print("\n[SERVER] Interrupted by user.")
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("Connection Terminated!")
#####################################

#####################################
if __name__ == '__main__':
    try:
        main()
    finally:
        print("[SYSTEM] Shutting down ROS...")

        node.destroy_node()
        rclpy.shutdown()

        thread.join()
#####################################

