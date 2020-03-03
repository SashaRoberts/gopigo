# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:27:18 2020

@author: nick
"""

import cv2
import os

def take_picture(image_test_folder):
    video_capture = cv2.VideoCapture(0)
    
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
        
    # Read picture. ret === True on success
    ret, frame = video_capture.read()
    
    # Close device
    video_capture.release()

    # write file as jpg
    cv2.imwrite(os.path.join(image_test_folder, 'robot_fov.jpg'), frame)
    
    return 