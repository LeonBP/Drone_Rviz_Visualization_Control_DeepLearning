# Drone_Rviz_Visualization_Control_DeepLearning
Based in a previous system available here: https://github.com/anacmurillo/unizar_interactive_robotics

Proyect that launches a Rviz window that shows the Drone parameters and allows to send movement commands from visual information

To launch this you need to have a Drone running (Tested with Rotors simulation). The topics that feed the visualizator are stored in the Rviz file. It can be open with Rviz to change the topics. The different topics needed and parameters used are defined in the Config.py Python script.

For the Full system to work you need to launch (in different terminals):

	-Python script Viz_Dron.py is to launch the a window that works under Rviz that gives information of the Drone, the possible movements and the ability to decide if the movement is correct or not (as a fail-safe).

	-Python script Mov_Ev.py is to process the camera output and obtain the direction the User is pointing and feed the Drone with it.

	-Python script Moves.py is to create the different visual helps for the first script.

The gesture recognition is done in two phases: person detection and gesture classification. The models used in each phase are set in the config file:

    -Variable detectorID is the index of the array detectors that contains the model info of the detector used in the first phase.
    
    -Variable classifierID is the index of the array classifiers that contains the model info of the classifier used in the second phase.
    
By default, both these variables are ser to 2. The models in those indexes are a Mask R-CNN [1] model and a MobileNet [2] fine-tuned for this purpose. It is necessary to have Caffe2, Detectron [3] and Tensoflow installed and the MobileNet model inside the model directory to be able to run the version of the developed system.

[1] From the set 12_2017_baselines, the trained model e2e_mask_rcnn_R-101-FPN_2x available from [3].
[2] Model pre-trained on ImageNet and fine-tuned for the gesture classification problem.
[3] Official implementation of Mask R-CNN: https://github.com/facebookresearch/Detectron
