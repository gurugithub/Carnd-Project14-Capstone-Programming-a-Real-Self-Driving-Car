#!/usr/bin/env python
import rospy
import tf
import cv2
import yaml
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# split tl_classifer. Double keying but better management
from light_classification.tl_classifier import TLClassifier
from light_classification.tl_classifier_carla import TLClassifierCarla

# additional imports
import math
import numpy as np
import os

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        tl_classifier_class = rospy.get_param('~model_class')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl_waypoints_idx = []

        self.harvest_images = False
        if self.harvest_images:
            if not (os.path.exists("./tl_images")):
                os.mkdir("./tl_images")
            self.debug_image_count = 0

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()
        self.light_classifier = globals()[tl_classifier_class]()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # ROS publishers
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint',
                                                      Int32, queue_size=1)
        self.log_pub = rospy.Publisher('/vehicle/visible_light_idx',
                                       Int32, queue_size=1)

        # ROS subscribers

	'''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        sub1 = rospy.Subscriber('/current_pose', PoseStamped,
                                self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane,
                                self.waypoints_cb, queue_size=1)

        # /vehicle/traffic_lights provides you with the location of the traffic
        # light in 3D map space and helps you acquire an accurate ground truth
        # data source for the traffic light classifier by sending the current
        # color state of all traffic lights in the simulator.
        # When testing on the vehicle, the color state will not be available.
        # You'll need to rely on the position of the light and the camera image
        # to predict it.
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                                self.traffic_cb, queue_size=1)

        # Set big enough buffer size for the image subscriber
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
        image_depth = 3  # RGB
        buffer_size_img = 2 * (image_width * image_height * image_depth)

        sub6 = rospy.Subscriber('/image_color', Image,
                                self.image_cb, queue_size=1,
                                buff_size=buffer_size_img)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes
            the index of the waypoint closest to the red light's stop line to
            /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp_idx, state = self.process_traffic_lights()

        if self.harvest_images:
            self.harvest_image(self.camera_image)

        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it.
        # Otherwise the previous stable state is used.
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp_idx = light_wp_idx if state == TrafficLight.RED else -1
            self.last_wp = light_wp_idx
            self.upcoming_red_light_pub.publish(Int32(light_wp_idx))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def harvest_image(self, image):
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        cv2.imwrite("./tl_images/image{}.jpg".format(
            self.debug_image_count), cv_image)
        self.debug_image_count += 1

    def distance(self, p1, p2):
        """
        Distance between two map coordinates copied from WaypointLoader class.
        """
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def get_closest_waypoint_idx(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        # Very high value is set as as initial distance.
        closest_waypoint_dist = 100000
        closest_waypoint_ind = -1

        # Looping through base waypoints to find the one closest to the car.
        for i in range(0, len(self.waypoints.waypoints)):
            waypoint_distance = self.distance(
                self.waypoints.waypoints[i].pose.pose.position,
                pose.position
            )
            if waypoint_distance < closest_waypoint_dist:
                # In case that closer waypoint has been found,
                # set new distance and new closest waypoint index.
                closest_waypoint_dist = waypoint_distance
                closest_waypoint_ind = i
        return closest_waypoint_ind

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color
                 (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        # Get image in OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Get classification
        output = self.light_classifier.get_classification(cv_image)

        return output  # light.state

    def get_tl_waypoints_idx(self):
        """ Converts array self.lights with trafic light positions to
            tl_waypoints_idx array with traffic light waypoint indexes
        """
        # List of positions that correspond to the line to stop in front of
        # for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        for stop_line_position in stop_line_positions:
            stop_line_position_pose = Pose()
            stop_line_position_pose.position.x = stop_line_position[0]
            stop_line_position_pose.position.y = stop_line_position[1]
            stop_line_position_pose.position.z = 0
            self.tl_waypoints_idx.append(
                self.get_closest_waypoint_idx(stop_line_position_pose)
            )

    def get_distance_in_track(self, a, b, track_length):
        """Finds the shortest signed distance between a and b,
            (b - a), considering the track_length

            Example 1
            a = car = 190
            b = light = 10
            track_length = 200
            output = 20 (in front of the car)

            Example 2
            a = car = 10
            b = light = 190
            track_length = 200
            output = -20 (behind the car)
        """

        output = b - a

        if (output < -0.5 * track_length):
            output += track_length
        elif (output > 0.5 * track_length):
            output -= track_length

        return output

    def get_closest_tl_wp_idx(self, car_pos_wp_idx, searching_dist_tl):
        """ Finds closest traffic light waypoint in searching range

        Args:
            car_pos_wp_idx (Integer): current position of the car
            searching_dist_tl: number of waypoints that will be searched
            for a traffic light

        Returns:
            light_number: number of the closest light, between 0 and N-1
                          where N is the number of lights
            light_wp_idx: waypoint index of closest traffic light in range
        """

        light_wp_idx = -1
        light_number = -1

        # In case that array with traffic light stop line waypoints
        # does not exist, create it
        if(self.tl_waypoints_idx == []):
            self.get_tl_waypoints_idx()
        get_distance_in_track = len(self.waypoints.waypoints)
        # Loop thorugh waypoints to find the closest traffic light waypoint
        smallest_tl_distance = 10000
        for i, tl_waypoint_idx in enumerate(self.tl_waypoints_idx):
            # Covered corner case where traffic light is just behind the
            # waypoint 0 and the car is near the last waypoint
            distance_between_wp = self.get_distance_in_track(
                    car_pos_wp_idx, tl_waypoint_idx, get_distance_in_track
                )
            if(distance_between_wp < searching_dist_tl and
                distance_between_wp < smallest_tl_distance and
                    distance_between_wp > 0):
                light_wp_idx = tl_waypoint_idx
                light_number = i
                smallest_tl_distance = distance_between_wp
        # Return index of closest traffic light waypoint in range
        return light_number, light_wp_idx

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists,
           and determines its location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a
                 traffic light (-1 if none exists)
            int: ID of traffic light color
                 (specified in styx_msgs/TrafficLight)

        """

        # searching_dist_tl parameter sets distance in which
        # traffic lights will be searched as number of waypoints
        searching_dist_tl = 200
        light_wp_idx = -1
        light_number = -1
        light = None

        if(self.pose and self.waypoints):
            car_pos_wp_idx = self.get_closest_waypoint_idx(self.pose.pose)
            light_number, light_wp_idx = self.get_closest_tl_wp_idx(
                car_pos_wp_idx, searching_dist_tl)

        # If waypoint has been found get traffic light state
        if light_wp_idx >= 0 and light_number >= 0 and \
           self.is_light_within_fov(self.lights[light_number]):
            light = self.lights[light_number]
            state = self.get_light_state(light)

            # Publish index of nearest visible traffic light (debug)
            self.log_pub.publish(light_number)

            rospy.loginfo("Traffic light detected, waypoint number: {}"
                          .format(light_wp_idx))
            return light_wp_idx, state

        return -1, TrafficLight.UNKNOWN

    def is_light_within_fov(self, light):
        """
        Determines whether the light is within the FOV of the camera

        Args:
            light (TrafficLight) - light to consider

        Returns: true if the light is within the FOV of the camera
        """
        # Get position of traffic light and ego
        light_pos = light.pose.pose.position
        ego_pos = self.pose.pose.position

        # Get ego heading
        q = self.pose.pose.orientation
        q_array = [q.x, q.y, q.z, q.w]
        _, _, ego_heading = tf.transformations.euler_from_quaternion(q_array)

        # Create vector from ego to light and compute its heading
        v_ego_to_light = [light_pos.x - ego_pos.x, light_pos.y - ego_pos.y]
        ego_to_light_heading = math.atan2(v_ego_to_light[1], v_ego_to_light[0])

        # Compute bearing to light
        bearing = self.angle_difference(ego_heading, ego_to_light_heading)

        # Check if it's within FOV
        return abs(bearing) < 0.5 * math.radians(40.0)

    @staticmethod
    def angle_difference(a, b):
        """
        Computes the shortest signed angle between two angles a and b

        Args:
            a (float) first angle
            b (float) second angle
        """
        output = b - a

        if (output < -math.pi):
            output += 2.0 * math.pi
        elif (output > math.pi):
            output -= 2.0 * math.pi

        return output


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
