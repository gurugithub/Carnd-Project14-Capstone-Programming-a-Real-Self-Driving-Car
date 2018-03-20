#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built)
the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed
linear and angular velocities.

You can subscribe to any other message that you find important or refer
to the document for list of messages subscribed to by the
reference implementation of this node.

One thing to keep in mind while building this node and the
`twist_controller` class is the status of `dbw_enabled`.
While in the simulator it's enabled all the time, in the real car that will
not be the case. This may cause your PID controller to accumulate error
because the car could temporarily be driven by a human
instead of your controller.

We have provided two launch files with this node.
Vehicle specific values (like vehicle_mass, wheel_base, etc)
should not be altered in these files.

We have also provided some reference implementations for PID controller
and other utility classes. You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it
on the various publishers that we have created in the `__init__` function.
'''


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        max_throttle_percent = rospy.get_param('~max_throttle_percent')

        # Create controller
        self.controller = Controller(vehicle_mass, decel_limit, accel_limit,
                                     wheel_radius, wheel_base,
                                     steer_ratio, max_lat_accel,
                                     max_steer_angle, brake_deadband,
                                     fuel_capacity, max_throttle_percent)
        self.velocity = None
        self.twist = None
        self.dbw_enabled = False

        self.last_dbw_enabled = False
        self.last_time = None

        self.last_action = ''

        # ROS publishers
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # ROS subscribers
        rospy.Subscriber('/current_velocity', TwistStamped,
                         self.velocity_cb, queue_size=1)
        rospy.Subscriber('/twist_cmd', TwistStamped,
                         self.twist_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool,
                         self.dbw_enabled_cb, queue_size=1)

        self.loop()

    def velocity_cb(self, velocity):
        self.velocity = velocity

    def twist_cb(self, twist):
        self.twist = twist

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data

    def has_valid_data(self):
        return self.twist is not None and \
               self.velocity is not None and \
               self.last_time is not None

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # Get current time
            now = rospy.get_rostime()

            # Publish only if DBW is enabled
            if self.has_valid_data() and self.dbw_enabled:
                # Reset the controller if necessary
                self.maybe_reset_controller()

                # Compute control commands
                diff = now - self.last_time
                throttle, brake, steer = self.controller.control(
                    self.twist, self.velocity, diff.to_sec())

                # Publish
                self.publish(throttle, brake, steer)

            # Update variables
            self.last_time = now
            self.last_dbw_enabled = self.dbw_enabled

            rate.sleep()

    def maybe_reset_controller(self):
        """
        Resets the controller if the DBW signal is re-activated
        """
        if self.dbw_enabled != self.last_dbw_enabled and \
           self.dbw_enabled:
            self.controller.reset()
            rospy.loginfo('Resetting Controller')

    def publish(self, throttle, brake, steer):
        # Create brake command
        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake

        # Create throttle command
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle

        # Create steering command
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer

        # Publish
        # NOTE: do not publish throttle and brake at the same time,
        # unless we switch from one to the other (to prevent the simulator
        # from keeping the last value)
        action = 'brake' if brake > 0.0 else 'throttle'

        if action != self.last_action:
            self.brake_pub.publish(bcmd)
            self.throttle_pub.publish(tcmd)

        elif action == 'brake':
            self.brake_pub.publish(bcmd)

        elif action == 'throttle':
            self.throttle_pub.publish(tcmd)

        self.steer_pub.publish(scmd)

        self.last_action = action


if __name__ == '__main__':
    DBWNode()
