#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('window_name', 'F1TENTH Camera')
        self.declare_parameter('auto_contrast', True)
        self.declare_parameter('brightness_gain', 2.0)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.auto_contrast = self.get_parameter('auto_contrast').get_parameter_value().bool_value
        self.brightness_gain = self.get_parameter('brightness_gain').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.frame_count = 0
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(f'Subscribed to {self.image_topic}')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_count += 1

            display = frame
            if self.auto_contrast:
                # Improve visibility when camera image is very dark.
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                y = self.clahe.apply(yuv[:, :, 0])
                if self.brightness_gain > 1.0:
                    y = cv2.convertScaleAbs(y, alpha=self.brightness_gain, beta=0)
                yuv[:, :, 0] = y
                display = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

            # Lightweight diagnostics to confirm image content is arriving.
            if self.frame_count % 10 == 0:
                min_v = int(frame.min())
                max_v = int(frame.max())
                mean_v = float(frame.mean())
                text = f'min:{min_v} max:{max_v} mean:{mean_v:.1f} gain:{self.brightness_gain:.1f}'
                cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(self.window_name, display)
            cv2.waitKey(1)
        except CvBridgeError as exc:
            self.get_logger().error(f'cv_bridge conversion error: {exc}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
