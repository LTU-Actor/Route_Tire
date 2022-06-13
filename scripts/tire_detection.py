#!/usr/bin/env python3
import encodings
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import imutils
from dynamic_reconfigure.server import Server
from Route_Tire.cfg import TireConfig

bridge = CvBridge()

tire_image_pub = rospy.Publisher('/tire_detection/tire_image', Image, queue_size=10)
tire_mask_pub = rospy.Publisher('/tire_detection/tire_mask', Image, queue_size=10)
tire_detect_pub = rospy.Publisher('/tire_detection/tire_detect', Bool, queue_size=10)
#cv_image = cv2.imread("/home/achoudhur/dbw_ws/src/my_polaris_pkg/photos/tire_1.jpg")

#cv_image = cv2.resize(cv_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
start_detection = False
start_time = 0
stopActor = False

dt = 0

def param_callback(config, level):
    global hue_low, hue_high, sat_low, sat_high, val_low, val_high
    hue_low = int(config.hue_l)
    hue_high = int(config.hue_h)
    sat_low = int(config.sat_l)
    sat_high = int(config.sat_h)
    val_low = int(config.val_l)
    val_high = int(config.val_h)
    return config


def readImage(ros_image):
    global bridge
    global start_detection, tire_image_pub, tire_mask_pub, tire_detect_pub, dt, start_time, stopActor
    
    #Grab image
    try:
       cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
       print(e)
       
    #cv_image = cv2.imread("/home/achoudhur/dbw_ws/src/my_polaris_pkg/photos/tire_1.jpg")
    
    #cv_image = cv2.resize(cv_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    
    mask = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2HSV)
    
    lower_thresh = np.array([hue_low, sat_low, val_low])
    upper_thresh = np.array([hue_high, sat_high, val_high])
    
    mask = cv2.inRange(mask, lower_thresh, upper_thresh)
    # mask = cv2.bitwise_not(mask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    image_cX = cv_image.shape[1] / 2
    image_cY = cv_image.shape[0] / 2
    
    #Mask Contour Detection
    contour_list_1 = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cntX, cntY = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            if abs(cntX - image_cX) < 100 and abs(cntY - image_cY) < 200:
                if area > 3000 and area < 25000:
                    contour_list_1.append(cnt)
    
    thresh_image = cv2.blur(cv_image, (3, 3))
    thresh_image = cv2.Canny(thresh_image, 100, 200)
    
    contours, hierarchy = cv2.findContours(thresh_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_cX = cv_image.shape[1] / 2
    image_cY = cv_image.shape[0] / 2
    center_tolerance = 150
    
    #Edge Detection contour detection
    contour_list_2 = []
    
    for contour in contours:
        """
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), False)
        if (len(approx) > 6 and len(approx) < 12):
            x, y = approx[0][0]
            if abs(x - image_cX) < center_tolerance and abs(x - image_cY) < 300: 
                contour_list_2.append(contour)
        """
        M = cv2.moments(contour)
        if M['m00'] != 0:
            x, y = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            if abs(x - image_cX) < center_tolerance and abs(y - image_cY) < center_tolerance:
                contour_list_2.append(contour)
    
    #Combine mask and edge detection contour lines by finding the contours that are close together    
    contour_list_3 = []
    for cnt in contour_list_2:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cntX, cntY = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            for contour in contour_list_1:
                Mmt = cv2.moments(contour)
                if Mmt['m00'] != 0:
                    contourX, contourY = int(Mmt['m10']/Mmt['m00']), int(Mmt['m01']/Mmt['m00'])
                    if (abs(cntX - contourX) < 50) and (abs(cntY - contourY) < 50):
                        contour_list_3.append(cnt)
                        break
    
    #Uncomment if you want to view individual contours for mask and edge detection respectively
    img1 = cv2.drawContours(cv_image, contour_list_1, -1, (150, 120, 255), 2)
    #img2 = cv2.drawContours(cv_image, contour_list_2, -1, (255, 50, 200), 2)
    
    
    
    if len(contour_list_1) != 0:
        if not start_detection: 
            start_time = rospy.Time.now()
            rospy.loginfo("Started Detecting tire...")
            dt = 0
            start_detection = True
            tire_detect_pub.publish(False)
        elif start_detection and dt < 2:
            dt = (rospy.Time.now() - start_time).to_sec()
            tire_detect_pub.publish(False)
        else:
            if not stopActor:
            	rospy.loginfo("Stopping Actor1...")
            	stopActor = True
            	tire_detect_pub.publish(True)
            
        #img3 = cv2.drawContours(cv_image, contour_list_3, -1, (50, 255, 255), 2)
    
        M = cv2.moments(contour_list_1[0])
        if M['m00'] != 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(cv_image, (cx,cy), 10, (100,255,255), -1)
            cv2.putText(cv_image, "Tire", (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1,(140, 255, 255),2,cv2.LINE_AA)
    else:
        start_detection = False
        stopActor = False
        tire_detect_pub.publish(False)
        
    
    tire_image_pub.publish(bridge.cv2_to_imgmsg(cv_image, encoding="bgr8"))
    tire_mask_pub.publish(bridge.cv2_to_imgmsg(mask))
    
    #cv2.imshow("Threshold Image", thresh_image)
    #cv2.imshow("Original Image", cv_image)
    #cv2.imshow("Inverted Threshold Image", mask)
    cv2.waitKey(3)
    

if __name__ == "__main__":
    rospy.init_node("tire_detection", anonymous=False)
    srv = Server(TireConfig, param_callback)
    imgtopic = rospy.get_param("~imgtopic_name")
    rospy.Subscriber(imgtopic, Image, readImage)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
     
