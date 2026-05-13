import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import sys
import termios
import tty
import select


class DriveControl(Node):

    def __init__(self):
        super().__init__('drive_control_keyboard')

        self.pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        # State
        self.speed = 0.0
        self.steering = 0.0

        # Limits (safe for SLAM)
        self.max_speed = 0.5
        self.max_steering = 0.5

        # Step increments
        self.speed_step = 0.05
        self.steer_step = 0.05

        # Timer loop
        self.timer = self.create_timer(0.05, self.loop)

        # Ensure we are in a real terminal
        if sys.stdin.isatty():
            self.settings = termios.tcgetattr(sys.stdin)
        else:
            self.get_logger().error("Run this node in a terminal (TTY).")
            rclpy.shutdown()
            return

        self.get_logger().info(
            "\nArrow Key Controls:\n"
            "  ↑ : increase speed\n"
            "  ↓ : decrease speed / reverse\n"
            "  ← : steer left\n"
            "  → : steer right\n"
            "  SPACE : stop\n"
            "  q : quit\n"
        )

    # 🔥 FIXED: Proper arrow key handling
    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)

        if rlist:
            key = sys.stdin.read(1)

            # Handle arrow keys (escape sequences)
            if key == '\x1b':
                key += sys.stdin.read(1)
                key += sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def loop(self):
        key = self.get_key()

        # Arrow keys
        if key == '\x1b[A':  # UP
            self.speed = min(self.speed + self.speed_step, self.max_speed)

        elif key == '\x1b[B':  # DOWN
            self.speed = max(self.speed - self.speed_step, -self.max_speed)

        elif key == '\x1b[D':  # LEFT
            self.steering = min(self.steering + self.steer_step, self.max_steering)

        elif key == '\x1b[C':  # RIGHT
            self.steering = max(self.steering - self.steer_step, -self.max_steering)

        elif key == ' ':
            self.speed = 0.0

        elif key == 'q':
            self.get_logger().info("Exiting teleop...")
            rclpy.shutdown()
            return

        # Publish command
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering

        self.pub.publish(msg)

        # Real-time feedback
        print(
            f"\rSpeed: {self.speed:.2f} m/s | Steering: {self.steering:.2f} rad",
            end='',
            flush=True
        )


def main():
    rclpy.init()
    node = DriveControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'settings'):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
