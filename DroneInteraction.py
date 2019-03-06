import math
import cv2 as cv
import numpy as np
import Detector
import os
import Classifier
import rospy
import roslib
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped,Point,PointStamped,Pose
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from Config import *


class VisualInteraction():
    def __init__(self):
        self.DETEC = Detector.Detector_one_person(" ")
        self.CLASSIF = Classifier.Classifier_directions()
        if self.CLASSIF.classifier['type'] not in self.DETEC.detector['type']:
            print "Detector and classifier have incompatible types."
            exit(1)

    def get_indication_from_images(self,images,scales):
        #start_time = time.time()
        results_d = self.DETEC.detect_pose(images,scales)
        #elapsed_time = time.time() - start_time
        #print("Elapsed detection time: %0.2f seconds." % elapsed_time)
        
        start_time = time.time()
        x,y,z, results_c = self.CLASSIF.classify_direction(results_d,right=True,coef_norm=np.max(images[0].shape))
        elapsed_time = time.time() - start_time
        print("Elapsed classification time: %0.2f seconds." % elapsed_time)
        results_d.update(results_c)
        return x,y,z, results_d

    def draw_skul(self,image,skul):
        c = self.DETEC.draw_skeleton(image,skul)
        return c

class Movement():
    def __init__(self,f):
        rospy.init_node('movement', anonymous=True)
        self.msg = PoseStamped()
        self.msg.header.frame_id=FRAME_ID_MOV
        self.pet_pub = rospy.Publisher("/Asking",String,queue_size = 1)
        self.pos_pub = rospy.Publisher("/suggested_movement", Marker,latch=True,queue_size = 1)
        self.dir_pub = rospy.Publisher(TOPIC_COMMAND, PoseStamped,queue_size = 1)
        if f:
            data = rospy.wait_for_message(TOPIC_ODOMETRY,Odometry, timeout=50.0).pose
            self.msg.pose=data.pose
            print self.msg
        
    def create_mark(self,x0,y0,z0,x,y,z,delete):
        marker = Marker()
        marker.header.frame_id = FRAME_ID_WORLD
        marker.type = 0
        marker.ns = 'posible'
        if not delete:
            marker.action = marker.ADD
        else:
            marker.action = 3
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        p1 = Point()
        p1.x =x0
        p1.y =y0
        p1.z =z0
        p2 = Point()
        p2.x =x
        p2.z =z
        p2.y =y
        marker.points=[p1,p2]
        return marker

    def mover(self,x,y,z):
        data = rospy.wait_for_message(TOPIC_ODOMETRY, Odometry, timeout=5.0).pose
        self.msg.pose.position=data.pose.position
        try:
            data = rospy.wait_for_message("/Output", String, timeout=0.01).data
        except :
            True
        marker = self.create_mark(self.msg.pose.position.x,self.msg.pose.position.y,self.msg.pose.position.z,self.msg.pose.position.x+x,self.msg.pose.position.y+y,self.msg.pose.position.z+z,False)
        self.pos_pub.publish(marker)
        data = ""
        while data != "Ok"and data != "No":
            data = rospy.wait_for_message("/Output", String, timeout=5000.0).data
            #data = "Ok"##
        if data != "No":
            marker = self.create_mark(self.msg.pose.position.x,self.msg.pose.position.y,self.msg.pose.position.z,self.msg.pose.position.x+x,self.msg.pose.position.y+y,self.msg.pose.position.z+z,True)
            self.msg.pose.position.x +=x
            self.msg.pose.position.y += y
            self.msg.pose.position.z += z
            if self.msg.pose.position.x >= X_MAX:
                self.msg.pose.position.x = X_MAX
            elif self.msg.pose.position.x <= X_MIN:
                self.msg.pose.position.x = X_MIN
            if self.msg.pose.position.y >= Y_MAX:
                self.msg.pose.position.y = Y_MAX
            elif self.msg.pose.position.y <= Y_MIN:
                self.msg.pose.position.y = Y_MIN
            if self.msg.pose.position.z >= Z_MAX:
                self.msg.pose.position.z = Z_MAX
            elif self.msg.pose.position.z <= Z_MIN:
                self.msg.pose.position.z = Z_MIN
            self.pos_pub.publish(marker)
            print "Action Acepted"
            self.dir_pub.publish(self.msg)
            return self.msg.pose.position.x,self.msg.pose.position.y,self.msg.pose.position.z
        else:
            print "Action Denied"
            marker = self.create_mark(self.msg.pose.position.x,self.msg.pose.position.y,self.msg.pose.position.z,self.msg.pose.position.x+x,self.msg.pose.position.y+y,self.msg.pose.position.z+z,True)
            self.pos_pub.publish(marker)
            return None,None,None
