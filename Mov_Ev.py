import math
import cv2 as cv
import numpy as np
import rospy
import roslib
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from DroneInteraction import VisualInteraction,Movement
from Config import *

global images
global count
#ImageCallback
def callback(data):
    global images
    global count
    if count is None:
        count=0
    if count ==0:
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(data, "bgr8")
            image = cv.resize(image, None, fx=RES_SCALE, fy=RES_SCALE)
            images.append((image,data.header.seq))
            if len(images)> N_IMAGES:
                images.pop(0)
        except CvBridgeError as e:
            print(e)
        count+=1
    else:
        count+=1
    if count >SKIP_FRAMES:
        count=0



# Main Loop
count=0
images = []
Video = VisualInteraction()
bridge = CvBridge()
image_sub = rospy.Subscriber(name=TOPIC_CAMERA,data_class=Image, callback=callback,queue_size=1)
Mov = Movement(True)
pet_pub = rospy.Publisher("/estimated_pose",Image,queue_size = 1)
f=open("moves.txt","w+")##
while not rospy.is_shutdown():
    while len(images) < N_IMAGES:
        True
        #print len(images)
    image_list = images[:N_IMAGES]
    images=images[N_IMAGES:]
    image_list = [i[0] for i in image_list]
    pet_pub.publish(bridge.cv2_to_imgmsg(image_list[N_IMAGES-1]))
    x_s, y_s, z_s, detected = Video.get_indication_from_images(image_list,SCALE)
    #z_s, y_s, x_s, detected = Video.get_indication_from_images(image_list,SCALE)
    if x_s is not None and y_s is not None and z_s is not None:
        if 'skeletons' in detected and len(detected['skeletons']) > 0:
            canvas1 = Video.draw_skul(image_list[N_IMAGES-1],detected['skeletons'][N_IMAGES-1])
            pet_pub.publish(bridge.cv2_to_imgmsg(canvas1))
        elif 'nobackgrounds' in detected and len(detected['nobackgrounds']) > 0:
            pet_pub.publish(bridge.cv2_to_imgmsg(detected['nobackgrounds'][N_IMAGES-1]))
        x_s,y_s,z_s = Transform_Dir(x_s,y_s,z_s)
        x,y,z = Mov.mover(x_s,y_s,z_s)
        f.write(detected['direction']+"\n")##
        if x is not None and y is not None and z is not None:
            data = rospy.wait_for_message(TOPIC_ODOMETRY, Odometry, timeout=5.0).pose.pose
            while (abs(data.position.x-x)>X_THRES or abs(data.position.y-y)>Y_THRES or abs(data.position.z-z)>Z_THRES):
                data = rospy.wait_for_message(TOPIC_ODOMETRY, Odometry, timeout=5.0).pose.pose
f.close()##


