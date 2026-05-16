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

        self.last_lane_center = 320
        self.last_leftx_base  = 320 - 150
        self.last_rightx_base = 320 + 150
        self.smooth_e = 0.0   # EMA-smoothed crosstrack error (m)

        # --- BEV Calibration Parameters ---
        # These are estimated for a 640x480 camera looking slightly down from z=0.12
        # You will need to tune these 4 source points to perfectly match a square on your track!
        # Widened to grab the absolute maximum width of the camera feed
        self.src_pts = np.float32([
            [30,  325],   # Bottom Left  — close to car, inside visible ROI
            [610, 325],   # Bottom Right — close to car, inside visible ROI
            [420, 248],   # Top Right    — ~2–3 m ahead, narrow
            [220, 248]    # Top Left     — ~2–3 m ahead, narrow
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

        # ROI — expose rows 235–420 (sky above, bumper/hood below)
        # This matches the BEV src_pts band (y=248 to y=415) so nothing is pre-blacked
        frame[:235, :] = 0   # Black out sky
        frame[330:, :] = 0   # Black out front bumper (calibrated value)

        # 3. Apply Perspective Transform (Bird's Eye View)
        warped = cv2.warpPerspective(frame, self.M, (640, 480))

        # 4. Color Thresholding — lowered for Gazebo gray lane markings
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)

        # 5. Sliding Window Lane Tracking
        # Step A: Find the base of the lines using CONSTRAINED search zones
        histogram = np.sum(binary[360:, :], axis=0) 
        
        # Near-full-width search zones — on a 90° turn a line can reach the image edge
        left_zone = histogram[10:320]
        right_zone = histogram[320:630]
        
        # If no pixels found in zone, default to MEMORY
        if np.max(left_zone) > 0:
            leftx_base = np.argmax(left_zone) + 10
            self.last_leftx_base = leftx_base
        else:
            leftx_base = self.last_leftx_base
            
        if np.max(right_zone) > 0:
            rightx_base = np.argmax(right_zone) + 320
            self.last_rightx_base = rightx_base
        else:
            rightx_base = self.last_rightx_base

        # Step B: Set up the sliding windows
        # 12 windows (40px each) vs 9 (53px each) — finer steps reduce angle-per-window
        # on 90° curves, keeping each window's line segment more vertical.
        nwindows = 12
        window_height = int(binary.shape[0] / nwindows)
        
        # margin=90: on a 90° curve a line can shift ~80–100px between window centres
        margin = 90
        minpix = 15   # slightly lower so windows lock on even with sparse pixels on curves

        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        out_img = np.dstack((binary, binary, binary))

        # Step C: Slide the windows up the screen, tracking positions at each level
        left_positions  = []   # x position of left line at each window (bottom=near, top=far)
        right_positions = []

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
            
            if len(good_left_inds) > minpix:
                new_leftx = int(np.mean(nonzerox[good_left_inds]))
                left_dx = new_leftx - leftx_current
                leftx_current = new_leftx
                
            if len(good_right_inds) > minpix:        
                new_rightx = int(np.mean(nonzerox[good_right_inds]))
                right_dx = new_rightx - rightx_current
                rightx_current = new_rightx
                
            # Curve Borrowing: if one line is lost, mirror the other line's shift
            if len(good_right_inds) <= minpix and len(good_left_inds) > minpix:
                rightx_current += left_dx
            elif len(good_left_inds) <= minpix and len(good_right_inds) > minpix:
                leftx_current += right_dx

            left_positions.append(leftx_current)
            right_positions.append(rightx_current)

        # Step D: Near-field crosstrack error (bottom 3 windows = where the car IS now)
        #         Far-field heading estimate (top 3 windows = where the lane IS going)
        #         Using near-field prevents the "early turning" bug where the
        #         detector reported the curve apex instead of the car's actual offset.
        camera_center = 320
        # N_NEAR=1: use only the single bottom-most window for crosstrack.
        # Camera geometry means even this window is ~0.6m ahead of the camera;
        # averaging 3 windows (N_NEAR=3) added another 0.3-0.5m of preview
        # and caused the car to steer into curves before reaching them.
        N_NEAR = 1
        near_left  = int(np.mean(left_positions[:N_NEAR]))
        near_right = int(np.mean(right_positions[:N_NEAR]))
        lane_center = int((near_left + near_right) / 2)
        
        error_px = camera_center - lane_center

        # Convert to metres  (track width ~0.8 m visible across ~340 px in BEV)
        meters_per_pixel = 0.8 / 340.0
        crosstrack_error_m = error_px * meters_per_pixel
        
        # Heading error: angle from near-center to far-center
        far_left   = int(np.mean(left_positions[-N_NEAR:]))
        far_right  = int(np.mean(right_positions[-N_NEAR:]))
        far_center = int((far_left + far_right) / 2)
        dx = far_center - lane_center
        dy = 240   # virtual BEV pixels between near and far band
        heading_error_rad = -math.atan2(dx, dy)

        # EMA on crosstrack: smooth out single-window noise before publishing
        # Weight 0.25 = ~100ms lag at 20Hz — removes high-freq jitter while
        # still responding to real lane changes within a few frames.
        self.smooth_e = 0.25 * crosstrack_error_m + 0.75 * self.smooth_e

        # --- Publish the Errors ---
        msg_e = Float32()
        msg_e.data = float(self.smooth_e)
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