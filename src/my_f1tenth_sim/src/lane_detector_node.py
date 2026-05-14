#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from std_msgs.msg import Float32

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')
        
        # Subscribe to the camera feed defined in your car.xacro
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        
        # --- Add these Publishers ---
        self.e_pub = self.create_publisher(Float32, '/perception/crosstrack_error', 10)
        self.th_pub = self.create_publisher(Float32, '/perception/heading_error', 10)

        self.last_lane_center = 320 # Start by assuming the center is straight ahead
        self.last_leftx_base = 320 - 150
        self.last_rightx_base = 320 + 150

        # --- BEV Calibration Parameters ---
        # These are estimated for a 640x480 camera looking slightly down from z=0.12
        # You will need to tune these 4 source points to perfectly match a square on your track!
        # Widened to grab the absolute maximum width of the camera feed
        self.src_pts = np.float32([
            [0, 330],    # Bottom Left: Pushed all the way to the left edge
            [640, 330],  # Bottom Right: Pushed all the way to the right edge
            [480, 260],  # Top Right: Widened and moved slightly up
            [160, 260]   # Top Left: Widened and moved slightly up
        ])
        
        # Map the source points to a flat top-down square
        self.dst_pts = np.float32([
            [0, 480],    # Bottom Left
            [640, 480],  # Bottom Right
            [640, 0],    # Top Right
            [0, 0]       # Top Left
        ])
        
        # Calculate the perspective warp matrix
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

    def image_callback(self, msg):
        # 1. Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 2. Aggressive ROI (Mask out sky and bumper)
        frame[:270, :] = 0   # Black out the Sky (Moved down to 270)
        frame[330:, :] = 0   # Black out the Bumper

        # 3. Apply Perspective Transform (Bird's Eye View)
        warped = cv2.warpPerspective(frame, self.M, (640, 480))

        # 4. Color Thresholding (Lowered significantly for Gazebo lighting)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

        # 5. Sliding Window Lane Tracking
        # Step A: Find the base of the lines using CONSTRAINED search zones
        histogram = np.sum(binary[360:, :], axis=0) 
        
        # Constrain search to ignore extreme outer edges
        left_zone = histogram[100:320]
        right_zone = histogram[320:540]
        
        # If no pixels found in zone, default to MEMORY
        if np.max(left_zone) > 0:
            leftx_base = np.argmax(left_zone) + 100
            self.last_leftx_base = leftx_base
        else:
            leftx_base = self.last_leftx_base
            
        if np.max(right_zone) > 0:
            rightx_base = np.argmax(right_zone) + 320
            self.last_rightx_base = rightx_base
        else:
            rightx_base = self.last_rightx_base

        # Step B: Set up the sliding windows
        nwindows = 9
        window_height = int(binary.shape[0] / nwindows)
        
        margin = 40       # DECREASED: Tighter boxes to avoid snagging horizontal lines
        minpix = 20
        max_shift = 40    # NEW: Maximum pixels a box can shift horizontally in one step

        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        out_img = np.dstack((binary, binary, binary))

        # Step C: Slide the windows up the screen
        for window in range(nwindows):
            win_y_low = binary.shape[0] - (window + 1) * window_height
            win_y_high = binary.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_dx = 0
            right_dx = 0
            
            # Calculate shift, but CLAMP it to prevent jumping across horizontal lines
            if len(good_left_inds) > minpix:
                new_leftx = int(np.mean(nonzerox[good_left_inds]))
                left_dx = new_leftx - leftx_current
                left_dx = max(-max_shift, min(max_shift, left_dx)) # Clamp the jump
                leftx_current += left_dx
                
            if len(good_right_inds) > minpix:        
                new_rightx = int(np.mean(nonzerox[good_right_inds]))
                right_dx = new_rightx - rightx_current
                right_dx = max(-max_shift, min(max_shift, right_dx)) # Clamp the jump
                rightx_current += right_dx
                
            # Curve Borrowing (Handles the dashed line gaps)
            if len(good_right_inds) <= minpix and len(good_left_inds) > minpix:
                rightx_current += left_dx
            elif len(good_left_inds) <= minpix and len(good_right_inds) > minpix:
                leftx_current += right_dx

        # Step D: Calculate the target based on where the top windows ended up
        # This gives us a target point ahead of the car on the curve
        camera_center = 320
        lane_center = int((leftx_current + rightx_current) / 2)
        
        error_px = camera_center - lane_center
        # self.get_logger().info(f"Crosstrack Error (px): {error_px}")

        # 1. Convert to Meters (Using your average of 340 px)
        meters_per_pixel = 0.8 / 340.0 
        crosstrack_error_m = error_px * meters_per_pixel
        
        # 2. Estimate Heading Error (Lookahead)
        lookahead_hist = np.sum(binary[150:200, margin:640-margin], axis=0)
        
        if np.max(lookahead_hist) > 0:
            lookahead_center = np.argmax(lookahead_hist) + margin
            dx = lookahead_center - lane_center
            dy = 100 
            heading_error_rad = -math.atan2(dx, dy)
        else:
            heading_error_rad = 0.0

        # --- Publish the Errors ---
        msg_e = Float32()
        msg_e.data = float(crosstrack_error_m)
        self.e_pub.publish(msg_e)

        msg_th = Float32()
        msg_th.data = float(heading_error_rad)
        self.th_pub.publish(msg_th)

        # Draw the target line (Red for car, Blue for target)
        cv2.line(out_img, (camera_center, 0), (camera_center, 480), (0, 0, 255), 2)
        cv2.line(out_img, (lane_center, 0), (lane_center, 480), (255, 0, 0), 2)

        # Show the processing steps in live windows
        cv2.imshow("Raw Feed (with Mask)", frame)
        cv2.imshow("Bird's Eye View", warped)
        cv2.imshow("Sliding Windows", out_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()