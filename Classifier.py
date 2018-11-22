"""

Contains a class Classifier_directions to load the model to be used as classifier.

"""
KERAS_MODELS = "model/"

import os
from config_reader import config_reader
import numpy as np
import math
from Config import *

from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto(intra_op_parallelism_threads=4,\
       inter_op_parallelism_threads=4, allow_soft_placement=True,\
       device_count = {'CPU' : 1, 'GPU' : 0})
session = tf.Session(config=config)
K.set_session(session)
from keras import __version__
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope # Mandatory to be able to load a MobileNet model
from keras.layers import DepthwiseConv2D, ReLU
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.mobilenet import relu6

class Classifier_directions:
    def __init__(self):
        self.param, _, self.classifier = config_reader()
        if self.classifier['model_file']!='heuristic':
            MODEL_PATH = os.path.join(KERAS_MODELS,self.classifier['model_file'])
            assert os.path.exists(MODEL_PATH)
            with CustomObjectScope({'relu6': ReLU(6.),'DepthwiseConv2D': DepthwiseConv2D}): # specific to MobileNet
                self.classifier_model = load_model(MODEL_PATH)
                self.current_streak = None
                self.new_streak()
    
    def reorder_input(self, part_str, input_order):
        '''
        Returns a list with the indexes of the elements in part_str that appear in input_order
        '''
        return [[j for j in range(len(part_str)) if input_order[i]==part_str[j]][0] for i in range(len(input_order))]
            
    def classify_direction(self,detected,right=True,coef_norm=None):
        '''
        Obtains a direction from the input data.
        
        Arguments:
            detected: Dictionary containing input data for the classifier.
            right: Whether to use the right hand or the left if the classifier is based on heuristics.
            coef_norm: Number used to normalize the skeleton in case this mode is used.
            
        Returns:
            x: Float containing the movement command in the X axis.
            y: Float containing the movement command in the Y axis.
            z: Float containing the movement command in the Z axis.
            results: Dictionary containing additional output data.
        '''
        results = {}
        if self.classifier['model_file']!='heuristic':
            skeletons = detected['skeletons']
            nobackgrounds = detected['nobackgrounds']
            x,y,z,direction = self.calc_model(skeletons,nobackgrounds,coef_norm)
            results['direction'] = direction
        else:
            for i in range(len(skeletons)):
                x,y,z,dir_r = self.calc_heuristic(skeleton[0],right)
            results['dir_r'] = dir_r
        return x,y,z, results
            
    def calc_model(self,skeletons,nobackgrounds,coef_norm=None):
        '''
        Obtains a direction from the input data.
        
        Arguments:
            skeletons: List of skeletons. A skeleton is a dictionary of keypoints that contain two coordinates. 
            nobackgrounds: List of images.
            coef_norm: Number used to normalize the skeleton in case this mode is used.
            
        Returns:
            x: Float containing the movement command in the X axis.
            y: Float containing the movement command in the Y axis.
            z: Float containing the movement command in the Z axis.
            direction: String identifying the direction recognized.
        '''
        X = None
        prediction=[]
        direction=None
        for i in range(max(len(skeletons),len(nobackgrounds))):
            if 'skeleton' in self.classifier['type']:
                skeleton = skeletons[i]
                # get the keys in the correct order
                keys_input = self.classifier['input_order']
                # calculate Euclidean Distance Matrix
                edm = np.zeros((len(keys_input),len(keys_input),1))
                for k1 in range(len(keys_input)):
                    for k2 in range(k1+1, len(keys_input)):
                        edm[k1, k2, 0] = edm[k2, k1, 0] = np.linalg.norm(np.array(skeleton[keys_input[k1]])-np.array(skeleton[keys_input[k2]]))
                # normalize the EDM
                if coef_norm != None:
                    edm = edm/float(coef_norm)
                else:
                    edm = edm/float(np.amax(edm))
                X = edm
                # do the inference
                if len(X.shape) < 4:
                    X = np.expand_dims(X, axis=0)
            elif 'nobackground' in self.classifier['type']:
                nobackground = nobackgrounds[i]
                X = nobackground
                # do the inference
                if len(X.shape) < 4:
                    X = np.expand_dims(X, axis=0)
                X = preprocess_input(X)
            inference = self.classifier_model.predict(X)[0]
            prediction = inference
            predicted_index = np.argmax(prediction)
            if prediction[predicted_index]>VOTE_THRESH:
                self.new_vote(predicted_index)
        print(prediction)
        print(self.current_streak)
        #{0: 'down', 1: 'downleft', 2: 'downright', 3: 'left', 4: 'right', 5: 'up', 6: 'upleft', 7: 'upright'}
        if self.current_streak[1]>CONSENSUS_MIN:
            direction = self.classifier['class_index'][self.current_streak[0]]
            self.new_streak()
        print(direction)
        x,y,z = self.directions_label(direction)
        return x,y,z,direction
    
    def new_streak(self):
        '''
        Initializes the streak list. The streak list has the following structure:
            [current_streak_id, current_streak_count, longest_streak_id, longest_streak_count, discrepancies_available]
        '''
        self.current_streak = [-1,0,-1,0,-1]
        
    def new_vote(self, i):
        '''
        Casts a new vote, updating the streak list.
        If the id of the vote is the same as the current_streak_id,
        the current_streak_count is incremented by 1.
        If it's not, if there are discrepancies_available, it is decreased by one.
        If discrepancies_available is 0, the current_streak_id is changed
        to the id of the vote and current_streak_count is resetted to 0.
        longest_streak_id and longest_streak_count always store the id and
        the count of the longes streak.
        
        [current_streak_id, current_streak_count, longest_streak_id, longest_streak_count, discrepancies_available]
        
        Arguments:
            i: id of the vote.
        '''
        if self.current_streak[0]==-1:
            self.current_streak = [i,1,i,1,N_DISCREPANCIES]
        elif self.current_streak[2]!=i:
            if self.current_streak[4]>0:
                self.current_streak[4] = self.current_streak[4] - 1
            else:
                self.current_streak[2] = i
                self.current_streak[3] = 1
                self.current_streak[4] = N_DISCREPANCIES
        else:
            self.current_streak[3] = self.current_streak[3] + 1
        if self.current_streak[3]>self.current_streak[1]:
            self.current_streak[0]=self.current_streak[2]
            self.current_streak[1]=self.current_streak[3]
        
    def directions_label(self,dir):
        '''
        Transaltes the direction label to direction command coordinates.
        
        Arguments:
            dir: Direction label.
            
        Returns:
            x: Float containing the movement command in the X axis.
            y: Float containing the movement command in the Y axis.
            z: Float containing the movement command in the Z axis.
        '''
        x=y=z=0.0
        if dir=='right':
            y = -1
        elif dir=='upright':
            y = -1
            z = 1
        elif dir=='up':
            z = 1
        elif dir=='upleft':
            y = 1
            z = 1
        elif dir=='downright':
            y = -1
            z = -1
        elif dir=='down':
            z = -1
        elif dir=='downleft':
            z = -1
            y = 1
        elif dir=='left':
            y = 1
        elif dir=='unknown':
            x=y=z=0.0
        else:
            x,y,z=None,None,None
        return x,y,z
        
        
##### LEGACY CODE

    def calc_heuristic(self,Skel,right):
        dir_l, dir_r = self.obtain_direction(Skel)
        if 'chest' in Skel:
            chest = [Skel['chest'][0],Skel['chest'][1]]
        else:
            chest = [np.mean((Skel['shoulder right'][0],Skel['shoulder left'][0])),
                    np.mean((Skel['shoulder right'][1],Skel['shoulder left'][1]))]
        dist_r = math.hypot(chest[0] - Skel['hand right'][0],
                            chest[1] - Skel['hand right'][1])
        dist_l = math.hypot(chest[0] - Skel['hand left'][0],
                            chest[1] - Skel['hand left'][1])
        if dist_r > dist_l or right:
            x,y,z=self.directions_angle(dir_r)
        else:
            x,y,z=self.directions_angle(dir_l)
        return x,y,z,dir_r

    def obtain_direction(self,skeleton):
        a_l, a_r= self.obtain_xy_angle(skeleton)
        z_l,z_r = self.obtain_z_angle(skeleton,a_l[2],a_r[2])
        return (a_l,z_l),(a_r,z_r)

    def obtain_xy_angle(self,skeleton):
        def plot_point(point, angle, length):
            # unpack the first point
            x, y = point

            # find the end point
            endy = y + length * math.sin(math.radians(angle))
            endx = x + length * math.cos(math.radians(angle))
            return (int(endx), int(endy))

        if 'hand left' in skeleton.keys() and 'arm left' in skeleton.keys():
            angle = np.rad2deg(math.atan2(skeleton['hand left'][0] - skeleton['arm left'][0],
                                          skeleton['hand left'][1] - skeleton['arm left'][1]))
            angle2 = np.rad2deg(math.atan2(skeleton['shoulder right'][0] - skeleton['shoulder left'][0],
                                          skeleton['shoulder right'][1] - skeleton['shoulder left'][1]))
            if angle2 <= 0.0:
                angle=angle-(180.0-abs(angle2))
            else:
                angle = angle + (180.0 - abs(angle2))
            if angle >= 0.0:
                angle = angle-180.0
            else:
                angle = 180.0+angle
            start = (skeleton['hand left'][1], skeleton['hand left'][0])
            end = plot_point(start, angle, 200)
            left= (start,end,angle)
        else:
            left = (None,None,None)
        if 'hand right' in skeleton.keys() and 'arm right' in skeleton.keys():
            angle = np.rad2deg(math.atan2(skeleton['hand right'][0] - skeleton['arm right'][0],
                                          skeleton['hand right'][1] - skeleton['arm right'][1]))
            angle2 = np.rad2deg(math.atan2(skeleton['shoulder right'][0] - skeleton['shoulder left'][0],
                                          skeleton['shoulder right'][1] - skeleton['shoulder left'][1]))
            if angle2 <= 0.0:
                angle=angle-(180.0-abs(angle2))
            else:
                angle = angle + (180.0 - abs(angle2))
            if angle >= 0.0:
                angle = angle-180.0
            else:
                angle = 180.0+angle
            start = (skeleton['hand right'][1], skeleton['hand right'][0])
            end = plot_point(start, angle, 200)
            right = (start, end,angle)
        else:
            right = (None,None,None)
        return left,right

    def obtain_z_angle(self,skeleton,xy_l,xy_r):
        if 'hand left' in skeleton.keys() and 'arm left' in skeleton.keys() and 'shoulder left' in skeleton.keys() and 'shoulder right' in skeleton.keys():
            gamma_l = 0.0
            dist = math.hypot(skeleton['shoulder left'][0] - skeleton['shoulder right'][0],
                              skeleton['shoulder left'][1] - skeleton['shoulder right'][1])
            dist_l = math.hypot(skeleton['arm left'][0] - skeleton['hand left'][0],
                                skeleton['arm left'][1] - skeleton['hand left'][1])
            if dist !=0:
                perc_l = float(dist_l) / float(dist) if (float(dist_l) / float(dist)) <= 1.0 else 1.0
                angle_l = 90.0 * (1.0+ gamma_l - (perc_l))
            else:
                angle_l = None
        else:
            angle_l = None
        if 'hand right' in skeleton.keys() and 'arm right' in skeleton.keys() and 'shoulder left' in skeleton.keys() and 'shoulder right' in skeleton.keys():
            gamma_r = 0.0
            dist = math.hypot(skeleton['shoulder left'][0] - skeleton['shoulder right'][0],
                              skeleton['shoulder left'][1] - skeleton['shoulder right'][1])
            dist_r = math.hypot(skeleton['arm right'][0] - skeleton['hand right'][0],
                                skeleton['arm right'][1] - skeleton['hand right'][1])
            if dist != 0:
                perc_r = float(dist_r) / float(dist) if (float(dist_r) / float(dist)) <= 1.0 else 1.0
                angle_r = 90.0 * (1.0+ gamma_r - (perc_r))
            else:
                angle_r = None
        else:
            angle_r = None
        return angle_l,angle_r
        
    def directions_angle(self,dir):
        a_r, z_r = dir
        x=y=z=0.0
        if z_r > 30.0:
            x = 1.0
        else:
            x = 0.0
        angle = a_r[2]
        if self.classifier['angle_limits'][0] <= angle <= self.classifier['angle_limits'][1]:
            y = -1
        elif self.classifier['angle_limits'][1] < angle <= self.classifier['angle_limits'][2]:
            y = -1
            z = 1
        elif self.classifier['angle_limits'][2] < angle <= self.classifier['angle_limits'][3]:
            z = 1
        elif self.classifier['angle_limits'][3] < angle <= self.classifier['angle_limits'][4]:
            y = 1
            z = 1
        elif self.classifier['angle_limits'][0] > angle >= self.classifier['angle_limits'][5]:
            y = -1
            z = -1
        elif self.classifier['angle_limits'][5] > angle >= self.classifier['angle_limits'][6]:
            z = -1
        elif self.classifier['angle_limits'][6] > angle >= self.classifier['angle_limits'][7]:
            z = -1
            y = 1
        else:
            y = 1
        return x,y,z

    def extract_hand(self,skel,dir,right):
        def plot_point(point, angle, length):
            # unpack the first point
            x, y = point

            # find the end point
            endx = x + length * math.sin(math.radians(angle))*-1.0
            endy = y + length * math.cos(math.radians(angle))*-1.0
            return (int(endy), int(endx))
        if right:
            v = np.array(skel.values())
            lenght =( min(v.max(0)[1]+30,480) - max(v.min(0)[1]-30,0) ) *0.1
            center = plot_point(skel["hand right"],dir[0][2],lenght)
            init = (int(center[0]-lenght),int(center[1]-lenght))
            finit = (int(center[0]+lenght), int(center[1]+lenght))
            return init,finit
        else:
            v = np.array(skel.values())
            lenght =( min(v.max(0)[1]+30,480) - max(v.min(0)[1]-30,0) ) *0.4
            center = plot_point(skel["hand left"],dir[0][2],lenght)
            init = (center[0] - lenght, center[1] - lenght)
            finit = (center[0] + lenght, center[1] + lenght)
            return init, finit

