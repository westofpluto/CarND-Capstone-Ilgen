########################################################
# This is the traffic light classifier.
# It is based on squeezenet
########################################################
import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import model_from_yaml
import rospy

from styx_msgs.msg import TrafficLight

CLASSIFIER_MODEL_DIR = 'light_classification/models/'
CLASSIFIER_MODEL_SIM_DIR = os.path.join(CLASSIFIER_MODEL_DIR,'sim')
CLASSIFIER_MODEL_SITE_DIR = os.path.join(CLASSIFIER_MODEL_DIR,'site')
CLASSIFER_MODEL_WEIGHTS_FILE = 'light_classification/classifier_model_weights.h5'
CLASSIFER_MODEL_YAML_FILE = 'light_classification/classifier_model.yaml'

def readFile(filename):
    handle=open(filename,'r')
    txt=handle.read()
    handle.close()
    return txt

class TLClassifier(object):
    def __init__(self,is_site):
        rospy.loginfo('[TL_CLASSIFIER] loading classifier, is_site is %s' % is_site)
        self.is_site=is_site
        if is_site:
            yaml_file = os.path.join(CLASSIFIER_MODEL_SITE_DIR,'classifier_model.yaml')
            weights_file = os.path.join(CLASSIFIER_MODEL_SITE_DIR,'classifier_model_weights.h5')
        else:
            yaml_file = os.path.join(CLASSIFIER_MODEL_SIM_DIR,'classifier_model.yaml')
            weights_file = os.path.join(CLASSIFIER_MODEL_SIM_DIR,'classifier_model_weights.h5')

        model_yaml = readFile(yaml_file)
        self.model = model_from_yaml(model_yaml)
        self.model.load_weights(weights_file)
        self.graph = tf.get_default_graph()

        #
        # we run classify_lights in a thread, so get_classification just returns the
        # most recent value we had for the classifier
        #
        self.classifier_result = TrafficLight.UNKNOWN
        rospy.loginfo('[TL_CLASSIFIER] Classifier is ready!')

    def get_classification(self,image):
        self.run_classifier(image)
        return self.classifier_result

    def run_classifier(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        predicted_light_state = -1
        with self.graph.as_default():
            resized_image = cv2.resize(image,(224,224))
            pred = self.model.predict(resized_image.reshape(1,224,224,3))[0]
            # pred is an array of 4 element
            predicted_light_state = np.argmax(pred)
            #rospy.logwarn("[CLASSIFIER]: %s" % str(pred))
            #rospy.logwarn("[CLASSIFIER]: %d" % predicted_light_state)

        #
        # In our trained model, we trained so that green images had a label index of 0,
        # yellow images had a label index of 1, red images had a label index of 2,
        # and unknown/none had a label index of 3
        #
        if predicted_light_state == 0:
            self.classifier_result = TrafficLight.GREEN
        elif predicted_light_state == 1:
            self.classifier_result = TrafficLight.YELLOW
        elif predicted_light_state == 2:
            self.classifier_result = TrafficLight.RED    
        else:
            self.classifier_result = TrafficLight.UNKNOWN

        #rospy.logwarn("[CLASSIFIER]: result: %d" % self.classifier_result)
        return self.classifier_result

