from configobj import ConfigObj
import numpy as np


def config_reader():
    config = ConfigObj('config')

    param = config['param']
    
    # Read the detection model and its parameters
    detector_id = param['detectorID']
    assert detector_id in config['detectors']
    detector = config['detectors'][detector_id]
    assert ((detector['load_mode'] == 'pb' and 'init' in detector and 'predict' in detector) or
        (detector['load_mode'] == 'pkl' and 'model_file' in detector and 'cfg_file' in detector))
    assert ('part_str' in detector and 'type' in detector)
    detector['part_str'] = np.array(detector['part_str'])
    if 'boxsize' in detector:
        detector['boxsize'] = int(detector['boxsize'])
    if 'stride' in detector:
        detector['stride'] = int(detector['stride'])
    if 'padValue' in detector:
        detector['padValue'] = int(detector['padValue'])
    
    # Read the classification model and its parameters
    classifier_id = param['classifierID']
    assert classifier_id in config['classifiers']
    classifier = config['classifiers'][classifier_id]
    assert 'model_file' in classifier
    if 'angle_limits' in classifier:
        classifier['angle_limits'] = np.array(classifier['angle_limits'])
        for i in range(len(classifier['angle_limits'])):
            classifier['angle_limits'][i] = int(classifier['angle_limits'][i])
    if 'input_order' in classifier:
        classifier['input_order'] = np.array(classifier['input_order'])
    if 'class_index' in classifier:
        classifier['class_index'] = np.array(classifier['class_index'])
    
    # Read and format other general parameters
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    param['octave'] = int(param['octave'])
    param['use_gpu'] = int(param['use_gpu'])
    assert param['use_gpu'] == 0 or param['use_gpu'] == 1
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = map(float, param['scale_search'])
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])
    assert param['GPUdeviceNumber'] >= 0

    return param, detector, classifier

if __name__ == "__main__":
    config_reader()
