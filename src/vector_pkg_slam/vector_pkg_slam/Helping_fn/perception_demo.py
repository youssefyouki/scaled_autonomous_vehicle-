#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO
import socket
import cv2

#####################################
# Load YOLO11-Pose model
model = YOLO('Models/yolo11n-pose.pt')  # uses COCO 17 keypoints :contentReference[oaicite:1]{index=1}
#####################################

#####################################
#Add your perception code in the function and please note that it is called in the while True so change inputs as you might need
def demo_Code(frame):
    #Initial Values for pwm [0,255], steer[75,105] 
    pwm= 255
    steer = 90 #zero steering
    #Add your code here!!
    
    return pwm, steer
#####################################
