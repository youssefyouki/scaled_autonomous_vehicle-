#!/usr/bin/env python3
"""
Terminal teleop for F1TENTH.

Controls:
    W - Move forward while held
    S - Move backward while held
    A - Turn left while held
    D - Turn right while held
    Space - Stop
    Q - Exit
"""

import os
import sys
import tty
import termios
import select
import fcntl
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist

try:
        from pynput import keyboard
except Exception:
        keyboard = None

class KeyboardDrive(Node):
    def __init__(self):
        super().__init__('keyboard_drive')

        self.lock_file = None
        self._acquire_single_instance_lock()

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Drive settings
        self.speed_step = 4.0
        self.steer_step = 0.3
        self.max_speed = 4.0
        self.max_steer = 0.5

        self.shutdown_requested = False
        self.held_keys = set()
        self.terminal_keys = {}
        self.listener_keys = set()
        self.fallback_timeout = 0.35
        self.input_mode = 'terminal'

        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.listener = None
        self._start_keyboard_listener()

        self.create_timer(0.02, self.publish_drive)

        self.get_logger().info('Keyboard drive node started.')
        self.get_logger().info('Hold W/S to drive and A/D to steer, Space to stop, Q to quit.')

    def _acquire_single_instance_lock(self):
        lock_path = '/tmp/my_f1tenth_keyboard_drive.lock'
        self.lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_file.write(str(os.getpid()))
            self.lock_file.flush()
        except OSError:
            self.get_logger().error('Another keyboard_drive instance is already running.')
            raise RuntimeError('keyboard_drive already running')

    def _start_keyboard_listener(self):
        if keyboard is None:
            self.get_logger().info('Using terminal input mode.')
            return

        try:
            self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.listener.start()
            self.input_mode = 'pynput'
            self.get_logger().info('Using pynput key press/release input.')
        except Exception as exc:
            self.listener = None
            self.input_mode = 'terminal'
            self.get_logger().warn(f'Falling back to terminal input: {exc}')

    def _on_press(self, key):
        if self.shutdown_requested:
            return

        if key == keyboard.Key.space:
            self.listener_keys.clear()
            return

        try:
            char = key.char.lower()
        except AttributeError:
            return

        if char == 'q':
            self.shutdown_requested = True
        elif char in 'wasd':
            self.listener_keys.add(char)

    def _on_release(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return

        self.listener_keys.discard(char)

    def poll_input(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            if key == 'q':
                self.shutdown_requested = True
            elif key == ' ':
                self.terminal_keys.clear()
            elif key in 'wasd':
                self.terminal_keys[key] = time.time()

    def publish_drive(self):
        if self.shutdown_requested:
            if rclpy.ok():
                self.get_logger().info('Exiting...')
                rclpy.shutdown()
            return

        self.poll_input()
        now = time.time()
        terminal_active_keys = {
            key for key, last_ts in self.terminal_keys.items()
            if (now - last_ts) < self.fallback_timeout
        }
        self.held_keys = terminal_active_keys | self.listener_keys

        speed = 0.0
        steering = 0.0

        if 'w' in self.held_keys:
            speed += self.speed_step
        if 's' in self.held_keys:
            speed -= self.speed_step
        if 'a' in self.held_keys:
            steering += self.steer_step
        if 'd' in self.held_keys:
            steering -= self.steer_step

        # Clamp values
        speed = max(-self.max_speed, min(self.max_speed, speed))
        steering = max(-self.max_steer, min(self.max_steer, steering))

        twist = Twist()
        twist.linear.x = float(speed)
        twist.angular.z = float(steering)
        self.cmd_vel_publisher.publish(twist)

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        try:
            if self.listener is not None:
                self.listener.stop()
        except Exception:
            pass
        if self.lock_file is not None:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
            except Exception:
                pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardDrive()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
