#!/usr/bin/env python

import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
import numpy as np
from geometry_msgs.msg import PointStamped,PoseStamped,Pose
from nav_msgs.msg import Odometry
from Config import *

topic = 'visualization_marker_array'
publisher = rospy.Publisher(topic, MarkerArray)
top = 'odometry_rect'
pub = rospy.Publisher(top, PoseStamped)
rospy.init_node('register')

MARKERS_MAX = 8

while not rospy.is_shutdown():
    markerArray = MarkerArray()
    data = rospy.wait_for_message(TOPIC_ODOMETRY, Odometry, timeout=5.0).pose
    for y in xrange(-1,2,1):
        for z in xrange(-1,2,1):
            pos=[z,y,0.0]
            if y == 0 and z == 0:
                marker = Marker()
                marker.header.frame_id = FRAME_ID_WORLD
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.4
                marker.color.a = 1.0
                marker.color.r = 0.6
                marker.color.g = 0.2
                marker.color.b = 0.7
                marker.pose.orientation = data.pose.orientation
                marker.pose.position.x = data.pose.position.x+pos[0]
                marker.pose.position.y = data.pose.position.y+pos[1]
                marker.pose.position.z = data.pose.position.z+pos[2]
            else:
                marker = Marker()
                marker.header.frame_id = FRAME_ID_WORLD
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.pose.orientation = data.pose.orientation
                marker.pose.position.x = data.pose.position.x+pos[0]
                marker.pose.position.y = data.pose.position.y+pos[1]
                marker.pose.position.z = data.pose.position.z+pos[2]

            # We add the new marker to the MarkerArray, removing the oldest
            # marker from it when necessary
            #if(count > MARKERS_MAX):
            #    markerArray.markers.pop(0)

            markerArray.markers.append(marker)
    orientation_q = data.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z,orientation_q.w]
    out = np.array(orientation_list)*np.array(ORIENTATION_LIST)

    data.pose.orientation.x = out[0]
    data.pose.orientation.y = out[1]
    data.pose.orientation.z = out[2]
    data.pose.orientation.w = out[3]

    # Renumber the marker IDs
    id = 0
    for m in markerArray.markers:
        m.id = id
        id += 1
    
    # Publish the MarkerArray
    publisher.publish(markerArray)

    Poses = PoseStamped()
    Poses.pose=data.pose
    Poses.header = marker.header
    pub.publish(Poses)
    #count += 1

    #rospy.sleep(0.01)
