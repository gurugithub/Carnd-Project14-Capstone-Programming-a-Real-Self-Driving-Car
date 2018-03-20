import rospy
import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight

MIN_CLASSIFICATION_CONFIDENCE = 0.85



class TLClassifierCarla(object):
    def __init__(self):
        # Load classifier

        self.graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open('light_classification/trained_models/udacity_bosch_mobilenet_20171016.pb', "rb") as f:
            graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def)

        input_name = "import/input"
        output_name = "import/final_result"

        # Get input tensor from the graph
        self.image_tensor = self.graph.get_operation_by_name(
            input_name).outputs[0]
        # Get classification tensor from the graph
        self.classification_tensor = self.graph.get_operation_by_name(
            output_name).outputs[0]

        with self.graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, [None, None, 3])
            float_caster = tf.cast(self.input_tensor, tf.float32)
            dims_expander = tf.expand_dims(float_caster, 0)
            resized = tf.image.resize_bilinear(dims_expander, [224, 224])
            self.image_normalized = tf.divide(tf.subtract(
                resized, [128]), [128])

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color
                 (specified in styx_msgs/TrafficLight)

        """

        with self.graph.as_default():
            # normalize image
            feed_dict = {self.input_tensor: image}
            image_np_expanded = self.sess.run(self.image_normalized,
                                              feed_dict=feed_dict)

            # Run inference
            feed_dict = {self.image_tensor: image_np_expanded}
            classes = self.sess.run(self.classification_tensor,
                                    feed_dict=feed_dict)

            results = np.squeeze(classes)

            output = self.graph_class_to_traffic_light(results)

            rospy.loginfo('Traffic Light: {}'
                          .format(self.traffic_light_to_str(output)))
            return output

    @staticmethod
    def graph_class_to_traffic_light(results):
        """ Converts from a class number as defined in the TensorFlow
            model, to a class number as defined in styx_msgs/TrafficLight
        """
        top_k = results.argsort()[:][::-1]

        rospy.loginfo("Traffic lights classes scores: {}".format(results))
        best_class = top_k[0]
        if results[best_class] < MIN_CLASSIFICATION_CONFIDENCE:
            rospy.loginfo("best class score {}".format(results[best_class]))
            return TrafficLight.UNKNOWN

        if best_class == 0:
            return TrafficLight.GREEN
        elif best_class == 1:
            return TrafficLight.UNKNOWN
        elif best_class == 2:
            return TrafficLight.RED
        elif best_class == 3:
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN

    @staticmethod
    def traffic_light_to_str(traffic_light):
        if traffic_light == TrafficLight.GREEN:
            return 'GREEN'
        elif traffic_light == TrafficLight.YELLOW:
            return 'YELLOW'
        elif traffic_light == TrafficLight.RED:
            return 'RED'
        return 'UNKNOWN'
