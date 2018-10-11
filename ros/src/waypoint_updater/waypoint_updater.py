#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightFullInfo
from scipy.spatial import KDTree
from copy import deepcopy
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number

RED_LIGHT = 0
YELLOW_LIGHT = 1
GREEN_LIGHT = 2

MPH_TO_MPS = 0.44704
#
# we only need this just in case we never get current car speed
# this should never happen
#
DEFAULT_SPEED_MPH = 4.0
DEFAULT_SPEED = DEFAULT_SPEED_MPH * MPH_TO_MPS

#
# max acceleration in MPS^2
#
MAX_ACCEL = 2.0

START_DECEL_TIME = 4.0
SAFE_STOP_TIME_YELLOW = 2.0
STOP_DIST = 3.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.loginfo("[WAYPOINT_UPDATER] init_node called")
        #
        # subscribe to pose and waypoints messages
        # Results in self.pose_cb called repeatedly, and self.waypoints_cb called once
        # Also subscribe to /current_velocity so we have complete state of car
        #
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.twist_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        #
        # simulator provides ground truth location and states of traffic lights
        # in the message '/vehicle/traffic_lights. This is not available in
        # the actual track. In either case, we need to subscribe to message
        # /traffic_waypoint, which is published by the traffic light detector node.
        # If we are in simulator mode, then this info will just be the ground truth
        # we get from '/vehicle/traffic_lights' (since in simulator mode we make
        # the traffic light detector just ue this directly). In live mode of course
        # we have to use an actual detector and publish detected results in that message.
        #
        # Note that we actually modify the /traffic_waypoint message to send back
        # both the state and the wayoint itself, and send it for the closest
        # traffic light ahead (not just closest red traffic light). This way we
        # can process yellow lights as well as red lights. 
        #
        rospy.Subscriber('/traffic_waypoint', TrafficLightFullInfo, self.traffic_cb)

        #
        # we don't do anything for now with /obstacle_waypoint
        #

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # 
        # initialize member vars
        #    self.MAX_SPEED = max allowed speed, get from waypoints. When we get 
        #        the waypoints initially, all waypoints have a velocity set to 
        #        max allowed velocity   
        #    self.pose = object with current position and orientation
        #    self.twist = object with current velocity and rotational velocity
        #    self.lane_header = header from lane message (lane message also contains 
        #                       base waypoints)
        #    self.base_waypoints = complete set of waypoints (sent once)
        #    self.waypoints_2d = list of lists, each inner list is x,y of waypoint
        #    self.waypoints_tree = instance of KDTree object made from waypoints_2d
        #    self.traffic_light_full_info = instance of TrafficLightFullInfo object
        #        which contains the nearest waypoint and current state of 
        #        nearest traffic light 
        #    self.output_waypoints = list of waypoints ahead of our car. This is what 
        #        we send out, ie the whole point of the updater is to send these
        #
        self.MAX_SPEED = None
        self.pose = None
        self.twist = None
        self.lane_header = None
        self.base_waypoints = None
        self.waypoints_2d = None 
        self.waypoints_tree = None
        self.traffic_light_full_info = None

        self.output_waypoints = None

        self.log_cnt = 0
        #
        # call self.loop to "run" the WaypointUpdater object with control of 
        # the publishing frequency
        #
        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_tree:
                #
                # Get closest waypoint, then compute output waypoints and publish
                #
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                #
                # set the output waypoints and the car velocity at these waypoints.
                # This will depend on what the car sees for traffic lights
                #
                self.set_output_waypoints_info(closest_waypoint_idx)
                #
                # Now we have the output waypoints in self.output_waypoints
                # So finally we publich them
                #
                self.publish_waypoints()

            rate.sleep()

    def get_closest_waypoint_idx(self):
        pos = self.pose.position
        x = pos.x
        y = pos.y
        closest_idx = self.waypoints_tree.query([x,y],1)[1]

        #
        # check if closest is ahead of or behind vehicle
        #
        closest_coord = self.waypoints_2d[closest_idx]
        if closest_idx > 0:
            prev_coord = self.waypoints_2d[closest_idx-1]
        else:
            n=len(self.waypoints_2d)
            prev_coord = self.waypoints_2d[n-1]

        #
        # hyperplane through closest_coords
        #
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect - prev_vect, pos_vect-cl_vect)
        if val > 0:
            closest_idx = (closest_idx+1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        lane = Lane()
        lane.header = self.lane_header
        lane.waypoints = self.output_waypoints 
        self.final_waypoints_pub.publish(lane)

    #
    # set_output_waypoints_info
    # This function is the workhorse of this node. It takes the input waypoint index
    # which determines where the car is, looks at current traffic light info, and
    # sets the output waypoints in self.output_waypoints
    # INPUTS:
    #    closest_waypoint_idx: index of closest waypoint
    # USES:
    #    self.base_waypoints, self.pose, self.twist, self.traffic_light_full_info
    # OUTPUT:
    #    sets self.output_waypoints
    #
    def set_output_waypoints_info(self,closest_waypoint_idx):
        #
        # get waypoints and handle wraparound
        #
        self.output_waypoints = deepcopy(self.base_waypoints[closest_waypoint_idx:closest_waypoint_idx+LOOKAHEAD_WPS])
        num_pts = len(self.output_waypoints)
        if num_pts < LOOKAHEAD_WPS:
            num_remaining = LOOKAHEAD_WPS - num_pts
            self.output_waypoints = self.output_waypoints + deepcopy(self.base_waypoints[0:num_remaining])

        #
        # Now get traffic light info and handle waypoint velocity
        #
        green_light_ahead = False
        yellow_light_ahead = False
        red_light_ahead = False
        unknown_light_ahead = False
        if self.traffic_light_full_info:
            if self.traffic_light_full_info.state == RED_LIGHT:
                red_light_ahead = True
            elif self.traffic_light_full_info.state == YELLOW_LIGHT:
                yellow_light_ahead = True
            elif self.traffic_light_full_info.state == GREEN_LIGHT:
                green_light_ahead = True
            else:
                unknown_light_ahead = True
        else:
            self.do_log("[WAYPOINT_UPDATER] self.traffic_light_full_info is None")

        if self.twist:
            car_speed = self.twist.linear.x
        else:
            car_speed = DEFAULT_SPEED    # need something to prevent exception

        #
        # handle red light ahead
        #
        if red_light_ahead:
            self.do_log("[WAYPOINT_UPDATER] RED light ahead")
            self.handle_light_ahead(closest_waypoint_idx,car_speed)

        elif yellow_light_ahead:
            #
            # for a yellow light, we must decide whether to stop at all, or to go thru the
            # yellow light. So, we first get the dustance to the yellow light to see
            # if we are close enough to just go through. If so, then we accelerate to go thru.
            # If not, then we take the same actions as if this were a red light
            #
            self.do_log("[WAYPOINT_UPDATER] YELLOW light ahead")
            dist_to_yellow_light = self.distance(self.base_waypoints, closest_waypoint_idx, self.traffic_light_full_info.waypoint_index)
            decision_dist = car_speed * SAFE_STOP_TIME_YELLOW
            if dist_to_yellow_light < decision_dist:
                #
                # the light is yellow but we are too close to it to stop, so let's
                # accelerate right through it.
                #
                self.go_forward_at_max_speed(closest_waypoint_idx,car_speed)
            else:
                #
                # the light is yellow and we are far enough away so we can stop for it.
                # So, we treat it like a red light and stop for it
                # We pass in the distance we just computed so we don't have to
                # compute it again
                #
                self.handle_light_ahead(closest_waypoint_idx,car_speed,distance=dist_to_yellow_light)
        else:
            #
            # no red or yellow light ahead, so just keep driving forward
            #
            if green_light_ahead:
                self.do_log("[WAYPOINT_UPDATER] GREEN light ahead")
            elif unknown_light_ahead:
                self.do_log("[WAYPOINT_UPDATER] UNKNOWN light ahead")

            self.go_forward_at_max_speed(closest_waypoint_idx,car_speed)

    #
    # drive_forward_at_max_speed
    # This sets the output_waypoints and speeds so that we accelerate to max speed
    #
    def go_forward_at_max_speed(self,closest_waypoint_idx,car_speed):
        is_first_wp = True
        prev_wp = None
        for wp in self.output_waypoints:
            if is_first_wp:
                speed=car_speed
                dist = self.distance_to_nearest_waypoint(closest_waypoint_idx)
                vf2 = speed*speed + 2.0*MAX_ACCEL*dist
                if vf2 <= 0.0:
                    vf = 0.0
                else:
                    vf = math.sqrt(vf2)
                if vf > self.MAX_SPEED:
                    vf = self.MAX_SPEED
                self.set_waypoint_velocity(wp,vf)
                prev_wp = wp
                is_first_wp = False
            else:
                speed=self.get_waypoint_velocity(wp)
                dist = self.distance_between_nearby_waypoints(prev_wp, wp)
                vf2 = speed*speed + 2.0*MAX_ACCEL*dist
                if vf2 <= 0.0:
                    vf = 0.0
                else:
                    vf = math.sqrt(vf2)
                if vf > self.MAX_SPEED:
                    vf = self.MAX_SPEED
                self.set_waypoint_velocity(wp,vf)
                prev_wp = wp

    #
    # handle_light_ahead
    # This handles the fact that there is a red or yellow light ahead that we will
    # want to stop for. If we detect that we are too far away from it to start the
    # deceleration process right now, then we just keep moving forward at max speed,
    # but if we see that we are close enough so that we want to start the deceleration
    # process, then we do that.
    #
    def handle_light_ahead(self,closest_waypoint_idx,car_speed,distance=None):
        if distance:
            dist_to_light = distance
        else:
            dist_to_light = self.distance(self.base_waypoints, closest_waypoint_idx, self.traffic_light_full_info.waypoint_index)
        start_decel_dist = car_speed * START_DECEL_TIME
        if dist_to_light > start_decel_dist:
            #
            # we are too far away from red light to stop now, so keep going
            # at maximum speed, just as if there were no red light
            #
            self.go_forward_at_max_speed(closest_waypoint_idx,car_speed)

        else:
            #
            # we are within range of the red/yellow light so we must start to decelerate
            #
            self.stop_for_light(closest_waypoint_idx,car_speed,dist_to_light)

    #
    # stop_for_light
    # This method executes the deceleration process to stop the car for the light ahead
    #
    def stop_for_light(self,closest_waypoint_idx,car_speed,dist_to_light):
        dist_to_stop_line = dist_to_light - STOP_DIST
        #
        # we could solve an optimal control problem here to choose a deceleration
        # profile that results in us being at zero speed at exactly the stop distance
        # ahead, but that's more complicated than we need. Instead, let's assume
        # a constant deceleration profile.
        #
        if dist_to_stop_line > 0.0:
            const_decel = (car_speed*car_speed)/(2.0*dist_to_stop_line)
            is_first_wp = True
            prev_wp = None
            for wp in self.output_waypoints:
                if is_first_wp:
                    speed=car_speed
                    dist = self.distance_to_nearest_waypoint(closest_waypoint_idx)
                    vf2 = car_speed*car_speed -2.0*const_decel*dist
                    if vf2 <= 0.0:
                        vf = 0.0
                    else:
                        vf = math.sqrt(vf2)
                    self.set_waypoint_velocity(wp,vf)
                    prev_wp = wp
                    is_first_wp = False
                else:
                    speed=self.get_waypoint_velocity(wp)
                    dist = self.distance_between_nearby_waypoints(prev_wp, wp)
                    vf2 = speed*speed - 2.0*const_decel*dist
                    if vf2 <= 0.0:
                        vf = 0.0
                    else:
                        vf = math.sqrt(vf2)
                    self.set_waypoint_velocity(wp,vf)
                    prev_wp = wp
        else:
            #
            # we have gone too far, we want all speeds to be zero
            #
            for wp in self.output_waypoints:
                self.set_waypoint_velocity(wp,0.0) 


    def pose_cb(self, msg):
        #
        # msg is an instance of PoseStamped which has the actual pose in msg.pose
        #
        self.pose = msg.pose 

    def twist_cb(self, msg):
        #
        # msg is an instance of TwistStamped which has the actual twist in msg.twist
        #
        self.twist = msg.twist 

    def waypoints_cb(self, lane_msg):
        #
        # store header and actual waypoints array from Lane message
        #
        rospy.loginfo("[WAYPOINT_UPDATER] waypoints_cb called")
        self.lane_header = lane_msg.header
        self.base_waypoints = lane_msg.waypoints
        if self.base_waypoints:
            rospy.loginfo("[WAYPOINT_UPDATER] got base_waypoints")
            self.MAX_SPEED = self.get_waypoint_velocity(self.base_waypoints[0])
            #
            # Just to be safe, use a MAX_SPEED slightly below the max allowed
            #
            self.MAX_SPEED = 0.85*self.MAX_SPEED

            if not self.waypoints_2d:
                #
                # each waypoint has a PoseStamped iteme pose and a TwistStamped item twist
                # so the actual Pose object is in waypoint pose.pose and the actual twist
                # object is in waypoint.twist.twist
                #
                self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.base_waypoints]
                numwp=len(self.waypoints_2d)
                rospy.loginfo("[WAYPOINT_UPDATER] got waypoints_2d with %d waypoints" % numwp)
                self.waypoints_tree = KDTree(self.waypoints_2d)
                rospy.loginfo("[WAYPOINT_UPDATER] got waypoints_tree ")

    def traffic_cb(self, msg):
        #
        # msg is an instance of TrafficLightFullInfo, which has a state 
        # and a waypoint_index
        #
        self.traffic_light_full_info = msg

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    #
    # This computes the total distance between two waypoints with two
    # different waypoint indices in self.waypoints
    #
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    #
    # This computes the distance between the car's current location and the nearest waypoint
    #
    def distance_to_nearest_waypoint(self,closest_waypoint_idx):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        dist = dl(self.pose.position, self.base_waypoints[closest_waypoint_idx].pose.pose.position)
        return dist

    #
    # This computes the distance between the two successive waypoints
    # where wp1 and wp2 are the actual waypoint objects (not the indices)
    #
    def distance_between_nearby_waypoints(self, wp1, wp2):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        dist = dl(wp1.pose.pose.position, wp2.pose.pose.position)
        return dist

    def do_log(self,msg):
        if self.log_cnt == 0:
            rospy.loginfo(msg)
        self.log_cnt += 1
        if self.log_cnt >= 10:
            self.log_cnt = 0

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
