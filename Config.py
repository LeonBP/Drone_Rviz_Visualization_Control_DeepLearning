

#######Frame_id for movement
#FRAME_ID_MOV = 'odom'
FRAME_ID_MOV = 'world'

#######Frame_id for the marker
#FRAME_ID = "vicon" #For Vicon Room
FRAME_ID_WORLD = "world"  #For Stereo Odometry and simulation

#######Topic for the command (PoseStamped type)
#TOPIC_COMMAND="/pegasus/command/pose" #Pegasus Drone
TOPIC_COMMAND="/firefly/command/pose" #Firefly(Simulation) Drone

#######Topic for the Odometry (Odometry msgs)
#TOPIC_ODOMETRY="/pegasus/vrpn_client/estimated_odometry" #Vicon Room
#TOPIC_ODOMETRY="/stereo_odometer/odometry" #Stereo Odometry
TOPIC_ODOMETRY="/firefly/odometry_sensor1/odometry" #Simulation

#######Limits for the movement of the drone 
X_MIN = -99.9 # 0.5 for the Vicon room
Y_MIN = -99.9 # -2.0 for the Vicon room
Z_MIN = 0.0 
X_MAX = 99.9 # 2.5 for the Vicon room
Y_MAX = 99.9 # 0.0 for the Vicon room
Z_MAX = 99.9

#######Number of frames skipped:
SKIP_FRAMES = 0

#######Scales used for the skeleton
SCALE=[0.7]

#######Scale for resizing before
RES_SCALE = 0.8

#######Topic for the frontal camera (compressed)
#TOPIC_CAMERA = "/camera/color/image_raw/compressed" #RealSense
TOPIC_CAMERA = "/camera/color/image_raw" #ROSBAG ETH Zurich
#TOPIC_CAMERA = "/videofile/image_raw" #VIDEOFILE
#TOPIC_CAMERA = "/camera/rgb/image_rect_color" #ROSBAG I3A
#TOPIC_CAMERA = "/firefly/camera/camera_frontal/image_raw/compressed" #Simulation

#######Number of images to process at a time
N_IMAGES= 1

#######Threshold to accept one vote from a frame
VOTE_THRESH=0.5

#######Number of (quasi)consecutive, equal votes to give an order
CONSENSUS_MIN= 4

#######Number of different votes allowed in the consensus
N_DISCREPANCIES= 1

#######Threshold to accept the position
X_THRES = 0.1
Y_THRES = 0.1
Z_THRES = 0.1

####Orientation Vector Transformation for Odometry
#ORIENTATION_LIST = [0,0,1.0,0] # Pegasus
ORIENTATION_LIST = [0,0,0,1.0] # Simulation

#######Rviz file
#RVIZ_FILE = "Visual_Real.rviz"
RVIZ_FILE = "Visual_Sim.rviz"

#######Function for transforming the output of the Visual Recogn to the Drone
def Transform_Dir(x,y,z):
	#y=y*-1  #Simulation
	return z,y,x

