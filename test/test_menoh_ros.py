#!/usr/bin/env python
import os
import sys
import unittest

import rospy
import rospkg
import rostest

from std_msgs.msg import String
from sensor_msgs.msg import Image

import cv2
import cv_bridge

class MenohRosTest(unittest.TestCase):
    def test_vgg16(self):
        rospy.loginfo("test_vgg16")
        rospack = rospkg.RosPack()
        pkg_root = rospack.get_path('menoh_ros')
        path = os.path.join(pkg_root, 'data', 'Light_sussex_hen.jpg')

        image = cv2.imread(path)
        bridge = cv_bridge.CvBridge()
        image_msg = bridge.cv2_to_imgmsg(image)

        pub = rospy.Publisher('/image_input/input', Image, queue_size=1)

        while pub.get_num_connections() == 0:
            if rospy.is_shutdown():
                self.fail("interrupted")
            rospy.loginfo("waiting connection")
            rospy.sleep(0.1)
        rospy.loginfo("connected")

        pub.publish(image_msg)

        result_msg = rospy.wait_for_message('/category_output/output', String)

        self.assertEqual(result_msg.data, 'n01514859 hen')


if __name__ == '__main__':
    #rostest.rosrun('menoh_ros', 'menoh_ros_test', MenohRosTest)
    rospy.init_node("test_vgg16")
    unittest.main()
