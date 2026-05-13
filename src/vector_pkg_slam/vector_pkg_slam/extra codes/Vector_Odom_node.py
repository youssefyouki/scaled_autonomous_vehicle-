from Define_Sensors import setup_sensors
from Helping_Functions import euler_to_quaternion, quaternion_to_euler
from Define_Subs_Pubs_Functions import publish_scan, publish_pointcloud
from Transformations import init_tf_broadcasters, publish_static_tf, publish_dynamic_tf

import rclpy
from geometry_msgs.msg import Point, Quaternion, Twist
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan, PointCloud2
from builtin_interfaces.msg import Time
from rclpy.clock import Clock

import numpy as np
import threading
import uuid
import time
#################################################
rclpy.init(args=None)
#namespace_name = f"Driver_{uuid.uuid4().hex[:8]}"
node = rclpy.create_node('Vector_Odom_Node')#,namespace=namespace_name)
#################################################

#################################################
# Spin in a separate thread
thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
thread.start()
#################################################

#################################################
#Initialize publisher and subscribers
pub_Odom = node.create_publisher(Odometry,'Car_Odom', 10)
pub_cmd = node.create_publisher(cmd_vel,'Car_Speed', 10)
#pub_Speed = node.create_publisher(AckermannDrive,'Car_Speed', 10)
#pub_Scan = node.create_publisher(LaserScan, '/LiDAR_scan', 10)
#pub_PC = node.create_publisher(PointCloud2, '/LiDAR_pointcloud', 10)


Vel = 0
steer = 0

def cmd_ackermann_callback(message):
 global Vel, steer
 
 Vel = message.speed
 steer = message.steering_angle

sub_Ack = node.create_subscription(AckermannDrive, 'cmd_ackermann', cmd_ackermann_callback, 1)
#################################################
    

#################################################
car = Driver()
timestep = int(car.getBasicTimeStep())

# Setup sensors
sensors = setup_sensors(car, timestep)
print(sensors)

##Call Sensors
#Localization Sensors
gps = sensors['gps']
imu = sensors['imu']
gyro = sensors['gyro']
#Perception Sensors
camera_r = sensors['camera_r']
camera_l = sensors['camera_l']
lidar = sensors['lidar']
radar = sensors['radar']
display = sensors['display']


# Base link initialization
init_tf_broadcasters(node)
if gps:
 car_position = gps.getValues()
if imu:
 headings = imu.getRollPitchYaw()
 headings_q = euler_to_quaternion(headings[0], headings[1], headings[2])
#publish_static_tf(node, 'map', 'odom', car_position, headings_q)
publish_static_tf(node, 'map', 'odom', (-45.0, 45.0, 0.4), (0.0, 0.0, 0.0, 1.0))
rclpy.spin_once(node, timeout_sec=1)
publish_static_tf(node, 'base_footprint', 'gps_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
rclpy.spin_once(node, timeout_sec=1)
publish_static_tf(node, 'base_footprint', 'imu_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
rclpy.spin_once(node, timeout_sec=1)
publish_static_tf(node, 'base_footprint', 'lidar_link', (1.11, 0.0, 1.16), (0.0, 0.0, 0.0, 1.0))
rclpy.spin_once(node, timeout_sec=1)
#################################################

#################################################
def main(args=None):
 try:
  #################################################
  tau = 0
  step_count = 0
  while car.step() != -1:
      st_time = time.time()
      curr_time = node.get_clock().now().to_msg()
      # Read localization sensors values here
      if gps:
          car_position = gps.getValues()
          car_speed = gps.getSpeed()
      if imu:
          roll, pitch, yaw = imu.getRollPitchYaw()
          car_yaw = yaw
          headings_q = euler_to_quaternion(roll, pitch, yaw)
          print(f"roll={roll}, pitch={pitch}, yaw={yaw}")
          print(headings_q)
          print(quaternion_to_euler(x=headings_q[0], y=headings_q[1], z=headings_q[2], w=headings_q[3]))
      if gyro:
          car_Rot_speed = gyro.getValues()
      
      # Odometry
      Car_Odom = Odometry()
      Car_Odom.header.stamp =  curr_time
      Car_Odom.header.frame_id = 'odom'
      Car_Odom.child_frame_id = 'base_footprint'
      
      Car_Odom.pose.pose.position = Point(x=car_position[0], y=car_position[1], z=car_position[2])
      Car_Odom.pose.pose.orientation = Quaternion(x=headings_q[0], y=headings_q[1], z=headings_q[2], w=headings_q[3])
      Car_Odom.twist.twist.linear.x = car_speed
      Car_Odom.twist.twist.angular.z = car_Rot_speed[2]
      pub_Odom.publish(Car_Odom)
      
      publish_dynamic_tf(node, 'odom', 'base_footprint', curr_time, (car_position[0], car_position[1], car_position[2]), (headings_q[0],headings_q[1],headings_q[2],headings_q[3]))
      
      msg_Ack = AckermannDrive(steering_angle = car.getSteeringAngle(), speed = car_speed)
      pub_Speed.publish(msg_Ack)
      
      
      # Read Perception sensors values here
      if camera_r:
          image_r = camera_r.getImage()
      if camera_l:
          image_l = camera_l.getImage()
      
      if lidar:
          #lidar_PC = publish_pointcloud(node,lidar)
          #pub_PC.publish(lidar_PC)
          #lidar_scan = publish_scan(node,lidar)
          #pub_Scan.publish(lidar_scan)
          #lidar_RI = lidar.getRangeImage()
          # inside main loop
          if step_count % 10 == 0:  # publish every 3 timestep
              #lidar_scan = publish_scan(node,lidar,curr_time)
              #pub_Scan.publish(lidar_scan)
              lidar_PC = publish_pointcloud(node,lidar,curr_time)
              pub_PC.publish(lidar_PC)
      """
      if radar:
          radar_Targets = radar.getTargets()
          
      try:
       print([radar_Targets[i].distance for i in range(len(radar_Targets))])
       print([radar_Targets[i].azimuth for i in range(len(radar_Targets))])
       print([(car_speed-radar_Targets[i].speed)*3.6 for i in range(len(radar_Targets))])
      except:
       print('no radar data')
      """
      update_display(display,car_position,car_speed)
      print(np.round(car_position,2), np.round(car_speed*3.6,2), np.round(car_yaw,2))
      
      car.setCruisingSpeed(Vel*3.6)
      car.setSteeringAngle(steer)
      end_time = time.time()
      tau  = end_time-st_time
      print(tau,(timestep/ 1000.0))
      if tau < (timestep/ 1000.0):
          print((timestep/ 1000.0) - tau)
          time.sleep((timestep/ 1000.0) - tau)
      end_time = time.time()
      tau  = end_time-st_time
      print(tau)
      step_count += 1
      print('---------------------')
  ################################################# 
 except KeyboardInterrupt:
  pass
#################################################

#################################################   
if __name__ == '__main__':
    main()
    node.destroy_node()
    rclpy.shutdown()
    thread.join()
#################################################
