import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):

    def __init__(self,vehicle_mass=None,fuel_capacity=None,brake_deadband=None,
                decel_limit=None, accel_limit=None, wheel_radius=None,
                wheel_base=None, steer_ratio=None,  max_lat_accel=None,
                max_steer_angle=None):

        min_speed = 0.0
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, 
                max_lat_accel, max_steer_angle)

        #
        # control gains
        #
        kp = 0.3
        ki = 0.1
        kd = 0.0
        min_throttle = 0.0
        max_throttle = 0.2
        self.throttle_controller = PID(kp,ki,kd,min_throttle,max_throttle)

        tau = 0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau,ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deabband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel,dbw_enabled,linear_vel,angular_vel):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        #
        # run LPF on velocity to filter out noise
        #
        current_vel = self.vel_lpf.filt(current_vel)

        #
        # get steering angle from yaw controller
        #
        steering = self.yaw_controller.get_steering(linear_vel,angular_vel,current_vel)

        #
        # now get throttle/brake signals
        #
        vel_error = linear_vel - current_vel
        self.last_vel=current_vel
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error,sample_time)
        brake = 0.0

        #
        # check to see if we are stopped and need to hold the car in place
        #
        if linear_vel == 0.0 and current_vel < 0.01:
            throttle = 0.0
            brake = 700.0

        #
        # check to see if we need to brake 
        #
        elif throttle < 0.1 and vel_error < 0.0:
            throttle = 0.0
            decel = max(vel_error,self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius

        #
        # return control signals
        #
        return throttle, brake, steering

