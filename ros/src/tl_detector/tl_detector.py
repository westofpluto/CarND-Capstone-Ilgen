#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, TrafficLightFullInfo
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import cv2
import yaml
import sys
import math
from scipy.spatial import KDTree

#
# import ML dependencies here
#
import numpy as np
import tensorflow as tf

STATE_COUNT_THRESHOLD = 1
#
# we do not bother trying to detect the state of any traffic
# light that is farther away than this distance
#
MAX_TRAFFIC_LIGHT_DETECTION_DISTANCE = 50.0

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.prev_pose = None
        self.pose = None
        self.velocity_unit_vector = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.has_image = False
        self.camera_image_msg = None
        self.lights = []
 
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']
        self.is_site = self.config['is_site']
        #
        # For the sim we can use ground truth traffic state data if we want to
        # For the actual site, we cannot use it even if we want to
        #
        if not self.is_site:
            self.use_ground_truth = self.config.get('use_ground_truth',False)
        else:
            self.use_ground_truth = False


        self.upcoming_traffic_light_pub = rospy.Publisher('/traffic_waypoint', TrafficLightFullInfo, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.is_site)

        self.state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        #
        # used just for sim for stats
        #
        self.num_correct_predictions=0
        self.num_incorrect_predictions=0
        self.num_predictions=0

        #rospy.spin()
        traffic_light_config = self.config['light_classifier']
        classifier_rate = rospy.Rate(traffic_light_config['rate'])
        while not rospy.is_shutdown():
            #
            # find and publish nearest traffic light info
            #
            self._find_publish_nearest_traffic_light_full_info()
            classifier_rate.sleep()

    def pose_cb(self, msg):
        self.prev_pose = self.pose
        self.pose = msg.pose
        if self.prev_pose and self.pose:
            try:
                self._compute_velocity_unit_vector()
            except Exception as ex:
                rospy.logwarn("[TL_DETECTOR] Exception: %s" % ex)

    def waypoints_cb(self, lane_msg):
        #rospy.loginfo("[TL_DETECTOR] waypoints_cb called")
        self.waypoints = lane_msg.waypoints
        numwp = len(self.waypoints)
        #rospy.loginfo("[TL_DETECTOR] found %d waypoints" % numwp)
        if self.waypoints:
            if not self.waypoints_2d:
                #
                # each waypoint has a PoseStamped iteme pose and a TwistStamped item twist
                # so the actual Pose object is in waypoint pose.pose and the actual twist
                # object is in waypoint.twist.twist
                #
                self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.waypoints]
                self.waypoints_tree = KDTree(self.waypoints_2d)


    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image_msg = msg

    def _find_publish_nearest_traffic_light_full_info(self):
        '''
        Publish upcoming traffic lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        light_wp, state = self._process_traffic_lights()
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_wp = light_wp
            self._publish_traffic_light_full_info(self.state,light_wp)
        else:
            self._publish_traffic_light_full_info(self.state,self.last_wp)
            self.state_count += 1

    #
    # This does the actual publishing
    #
    def _publish_traffic_light_full_info(self,state,light_waypoint_index):
        msg = TrafficLightFullInfo()
        msg.state = state
        msg.waypoint_index = light_waypoint_index
        self.upcoming_traffic_light_pub.publish(msg)

    def _get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            idx is int: index of the closest waypoint in self.waypoints

        """
        pos = pose.position
        x = pos.x
        y = pos.y
        closest_idx = self.waypoints_tree.query([x,y],1)[1]

        return closest_idx

    def _get_closest_light_ahead(self, pose):
        """Identifies the closest light to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest light ahead in self.lights

        """
        #
        # we have very few lights in either the simulation or the live test,
        # so it is easiest just to loop thru them rather than use KDTree
        #
        pos = pose.position
        x = pos.x
        y = pos.y
        closest_idx = -1
        closest_dist2 = None
        idx = 0
        for light in self.lights:
            xl = light.pose.pose.position.x
            yl = light.pose.pose.position.y

            #
            # make sure light is ahead, otherwise ignore it
            # we can only do this if the car velocity is nonzero
            #
            skip_light = False
            if self.velocity_unit_vector:
                dx = xl - x
                dy = yl - y
                car_to_light = [dx,dy]
                val = self.dot2d(car_to_light,self.velocity_unit_vector)
                if val < 0.0:
                    #
                    # light is behind us so continue
                    #
                    skip_light = True

            if not skip_light:
                if closest_dist2 is None:
                    closest_idx = idx
                    closest_dist2 = (x-xl)*(x-xl) + (y-yl)*(y-yl)
                else:
                    dist2 = (x-xl)*(x-xl) + (y-yl)*(y-yl)
                    if dist2 < closest_dist2:
                        closest_idx = idx
                        closest_dist2 = dist2
            idx+=1
        
        return closest_idx


    def _get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            return False

        #
        # we want to convert the image from internal ROS format to
        # CV2 GBR8 format because we trained our classifier model using GBR8 images.
        # The original raining images were in PNG and JPG format and in folders
        # that make it look like they are RGB, but we must remember that the training
        # loads the images using cv2.imread() which loads the images into BGR8
        # format, NOT RGB8!
        #
        cv2_image_bgr = self.bridge.imgmsg_to_cv2(self.camera_image_msg, "bgr8")

        #
        #Get classification
        #
        return self.light_classifier.get_classification(cv2_image_bgr)

    def _process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #
        # handle bad data just in case
        #
        if self.waypoints is None or self.waypoints_tree is None:
            #rospy.logwarn("[TL_DETECTOR]: No waypoints available")
            return -1, TrafficLight.UNKNOWN

        if self.pose is None:
            #rospy.logwarn("[TL_DETECTOR]: car pose is None")
            return -1, TrafficLight.UNKNOWN

        if not self.use_ground_truth and not self.has_image:
            #rospy.logwarn("[TL_DETECTOR]: no image available for detection")
            return -1, TrafficLight.UNKNOWN

        if not self.lights:
            #rospy.logwarn("[TL_DETECTOR]: no lights found")
            return -1, TrafficLight.UNKNOWN

        ############################################
        # no errors, so real processing begins here
        ############################################
        #
        # find the nearest traffic light ahead
        #
        light = None
        light_idx = self._get_closest_light_ahead(self.pose)
        #
        # handle case where there is no light ahead at any distance
        #
        if light_idx < 0:
            return -1, TrafficLight.UNKNOWN

        light = self.lights[light_idx]
        light_pos = light.pose.pose.position

        #
        # don't worry about the light if it is too far away
        #
        dist_to_light = self._find_distance(self.pose.position.x,
                            self.pose.position.y,
                            light.pose.pose.position.x,
                            light.pose.pose.position.y)

        if dist_to_light > MAX_TRAFFIC_LIGHT_DETECTION_DISTANCE:
            return -1, TrafficLight.UNKNOWN

        
        #
        # get stop line for light
        #
        stop_line_for_light = self.stop_line_positions[light_idx]

        #
        # get closest waypoints to the stop line for this light
        #
        stop_line_pose = Pose()
        stop_line_pose.position.x = stop_line_for_light[0]
        stop_line_pose.position.y = stop_line_for_light[1]
        stop_line_waypoint_idx = self._get_closest_waypoint(stop_line_pose)

        #
        # get closest waypoint to the car
        #
        car_waypoint_idx = self._get_closest_waypoint(self.pose)

        #
        # get light state
        #
        state = TrafficLight.UNKNOWN 
        if self.use_ground_truth:
            #rospy.logwarn("GROUND TRUTH")
            state = light.state
        else:
            #rospy.logwarn("CLASSIFYING IMAGE")
            state = self._get_light_state(light) 

            #
            # test against gt_state in sim
            #
            if not self.is_site:
                gt_state = light.state
                pred_state = state
                if gt_state == pred_state:
                    self.num_correct_predictions += 1
                else:
                    self.num_incorrect_predictions += 1
                self.num_predictions += 1

                if self.num_predictions > 0 and self.num_predictions % 10 == 0:
                    pred_acc = 100.0*float(self.num_correct_predictions)/float(self.num_predictions) 
                    rospy.logwarn('[TL_DETECTOR] prediction accuracy is %f percent' % pred_acc)

        #
        # return waypoint index of stopline and the light state
        #
        return stop_line_waypoint_idx, state

    #
    # computes the car velocity unit vector in world coordiantes.
    # This is a proxy for determining the car's orientation in
    # world coordinates.
    # If the velocity is zero, it does not update what is already there.
    #
    def _compute_velocity_unit_vector(self):
        px = self.prev_pose.position.x
        py = self.prev_pose.position.y
        x = self.pose.position.x
        y = self.pose.position.y
        dx = x - px
        dy = y - py
        vmag2 = dx*dx + dy*dy
        if vmag2 <= 0.0:
            vmag = 0.0
        else:
            vmag = math.sqrt(vmag2)
            self.velocity_unit_vector = [dx/vmag, dy/vmag]
            
    def _find_distance(self,x1,y1,x2,y2):
        return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

    def dot2d(self, v1, v2):
        return v1[0]*v2[0]+v1[1]*v2[1]

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
