#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import imutils
from dynamic_reconfigure.server import Server
from Route_Tire.cfg import ConeConfig

bridge = CvBridge()
detect_cone = rospy.Publisher('/cone_detection/cone_visible', Bool, queue_size=10)
cone_image_pub = rospy.Publisher('/cone_detection/cone_image', Image, queue_size=10)

start_time = 0
start_detection = False
displayLog = True
dt = 0


def param_callback(config, level):
    global hue_low, hue_high, sat_low, sat_high, val_low, val_high, color_threshold
    hue_low = int(config.hue_l)
    hue_high = int(config.hue_h)
    sat_low = int(config.sat_l)
    sat_high = int(config.sat_h)
    val_low = int(config.val_l)
    val_high = int(config.val_h)
    color_threshold = int(config.color_threshold)
    return config
    
    
def image_callback(ros_image):
    global start_detection, start_time, detect_cone, cone_image_pub, dt, displayLog
    
    try:
       cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
       print(e)
    
    cropped = cv_image[int((cv_image.shape[0] * 0.25)):int((cv_image.shape[0] * 0.75)),int((cv_image.shape[1] * 0.25)):int((cv_image.shape[1] * 0.75))]
    bound_box = cv2.rectangle(cv_image, (cv_image.shape[1] * 0.25, cv_image.shape[0] * 0.25), (cv_image.shape[1] * 0.75, cv_image.shape[0] * 0.75), (255,0,0), 2)
    
    #cv_image = cv2.resize(cv_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    mask = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2HSV)
    
    lower_thresh = np.array([hue_low, sat_low, val_low])
    upper_thresh = np.array([hue_high, sat_high, val_high])
    
    mask = cv2.inRange(mask, lower_thresh, upper_thresh)
    
    color_percentage = (np.sum(mask == 255) / (mask.shape[0] * mask.shape[1])) * 1000
    if color_percentage > color_threshold:
        if not start_detection:
            start_time = rospy.Time.now()
            rospy.loginfo("Detecting Cone...")
            start_detection = True
            detect_cone.publish(False)
            displayLog = True
            dt = 0
            detect_cone.publish(False)
        elif start_detection and dt < 0.25:
            dt = (rospy.Time.now() - start_time).to_sec()
            detect_cone.publish(False)
        else:
            if displayLog:
            	rospy.loginfo("Cone Detection Successful...")
            	displayLog = False
            detect_cone.publish(True)
    else:
    	start_detection = False
    	
    cone_image_pub.publish(bridge.cv2_to_imgmsg(mask))
    
if __name__ == "__main__":
    rospy.init_node("cone_detection", anonymous=False)
    srv = Server(ConeConfig, param_callback)
    imgtopic = rospy.get_param("~imgtopic")
    rospy.Subscriber(imgtopic, Image, image_callback)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
