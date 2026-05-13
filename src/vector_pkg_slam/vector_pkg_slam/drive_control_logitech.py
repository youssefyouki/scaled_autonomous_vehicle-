#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

import threading
from evdev import InputDevice, ecodes, list_devices


class DriveControlLogitech(Node):

    def __init__(self):
        super().__init__('drive_control_logitech')

        # Publisher (IDENTICAL)
        self.pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        # State (IDENTICAL)
        self.speed = 0.0
        self.steering = 0.0

        # Limits (IDENTICAL)
        self.max_speed = 0.5
        self.max_steering = 0.5

        # ==============================
        # 🔍 Find Logitech device
        # ==============================
        self.device = self.find_device()

        # ==============================
        # 📊 Axis ranges
        # ==============================
        abs_x = self.device.absinfo(ecodes.ABS_X)
        abs_t = self.device.absinfo(ecodes.ABS_Y)
        abs_b = self.device.absinfo(ecodes.ABS_Z)

        self.MIN_X, self.MAX_X = abs_x.min, abs_x.max
        self.MIN_T, self.MAX_T = abs_t.max, abs_t.min
        self.MIN_B, self.MAX_B = abs_b.max, abs_b.min

        # Shared raw values
        self.raw_steer = 0
        self.raw_throttle = 0
        self.raw_brake = 0

        # Start input thread
        threading.Thread(target=self.read_inputs, daemon=True).start()

        # Timer loop (IDENTICAL)
        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info("Logitech drive control started.")

    # ==============================
    # 🔍 Find device
    # ==============================
    def find_device(self):
        devices = [InputDevice(path) for path in list_devices()]
        for dev in devices:
            if "Driving" in dev.name or "Force" in dev.name or "GT" in dev.name:
                self.get_logger().info(f"Using: {dev.name} ({dev.path})")
                return dev

        self.get_logger().error("Logitech wheel not found.")
        rclpy.shutdown()

    # ==============================
    # 🎯 Scaling functions
    # ==============================
    def scale_steering(self, raw):
        norm = (raw - self.MIN_X) / (self.MAX_X - self.MIN_X)
        return (norm - 0.5) * 2 * self.max_steering  # → [-0.5, 0.5]

    def scale_pedal(self, raw, min_val, max_val):
        return (raw - min_val) / (max_val - min_val)

    # ==============================
    # 🔄 Input thread
    # ==============================
    def read_inputs(self):
        for event in self.device.read_loop():

            if event.type == ecodes.EV_ABS:

                if event.code == ecodes.ABS_X:
                    self.raw_steer = event.value

                elif event.code == ecodes.ABS_Y:
                    self.raw_throttle = event.value

                elif event.code == ecodes.ABS_Z:
                    self.raw_brake = event.value

    # ==============================
    # 🔁 Main loop (IDENTICAL LOGIC)
    # ==============================
    def loop(self):

        # Convert inputs
        steering = self.scale_steering(self.raw_steer)

        throttle = self.scale_pedal(
            self.raw_throttle,
            self.MIN_T,
            self.MAX_T
        )

        brake = self.scale_pedal(
            self.raw_brake,
            self.MIN_B,
            self.MAX_B
        )

        # Speed logic (forward - brake)
        speed = (throttle - brake) * self.max_speed

        # Clamp (safety)
        self.speed = max(-self.max_speed, min(self.max_speed, speed))
        self.steering = max(-self.max_steering, min(self.max_steering, steering))

        # Publish (IDENTICAL)
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering

        self.pub.publish(msg)

        # Feedback (IDENTICAL STYLE)
        print(
            f"\rSpeed: {self.speed:.2f} m/s | Steering: {self.steering:.2f} rad",
            end='',
            flush=True
        )


def main():
    rclpy.init()
    node = DriveControlLogitech()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
