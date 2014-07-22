import cv2
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image, Range

from baxter_core_msgs.msg import ITBState
from baxter_interface import (
    RobotEnable,
    AnalogIO, 
    DigitalIO,
    Gripper
)

from baxter_ikhelper import IKHelper

####################################################################################################

class BaxterNode(object):
    def __init__(self, camera_averaging=False):
        self._cvbr = CvBridge()

        self.rs = RobotEnable()
        self.ik = IKHelper()

        camera_topic = '/cameras/{0}_hand_camera/image_rect'
        if camera_averaging:
            camera_topic += '_avg'

        self.left_img = None
        self._left_camera_sub = rospy.Subscriber(
            camera_topic.format('left'), 
            Image,
            self._on_left_imagemsg_received)

        self.right_img = None
        self._right_camera_sub = rospy.Subscriber(
            camera_topic.format('right'), 
            Image,
            self._on_right_imagemsg_received)

        self.left_itb = None
        self._left_itb_sub = rospy.Subscriber(
            '/robot/itb/left_itb/state',
            ITBState,
            self._on_left_itbmsg_received)

        self.right_itb = None
        self._right_itb_sub = rospy.Subscriber(
            '/robot/itb/right_itb/state',
            ITBState,
            self._on_right_itbmsg_received)

        self.left_gripper = Gripper('left')
        self.right_gripper = Gripper('right')

        self._display_pub = rospy.Publisher(
            '/robot/xdisplay', 
            Image, 
            tcp_nodelay=True,
            latch=True)

    ############################################################################

    def start(self, spin=False, calibrate=False):
        self.rs.enable()

        # Wait for initial topic messages to come in.
        while self.left_img is None or \
              self.right_img is None or \
              self.left_itb is None or \
              self.right_itb is None:
            rospy.sleep(100)

        # Calibrate both grippers.
        if calibrate:
            self.left_gripper.calibrate()
            self.right_gripper.calibrate()

        if spin:
            rospy.spin()

    def display_image(self, img):
        img = cv2.resize(img, (1024, 600))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_msg = self._cv2_to_display_imgmsg(img)
        self._display_pub.publish(img_msg)

    ############################################################################

    def on_left_image_received(self, img):
        pass

    def on_right_image_received(self, img):
        pass

    def on_left_itb_received(self, itb):
        pass

    def on_right_itb_received(self, itb):
        pass

    ############################################################################

    def _camera_imgmsg_to_cv2(self, img_msg):
        return self._cvbr.imgmsg_to_cv2(img_msg, 'bgr8')

    def _cv2_to_display_imgmsg(self, img_msg):
        return self._cvbr.cv2_to_imgmsg(img_msg, 'bgr8')

    ############################################################################

    def _on_left_imagemsg_received(self, img_msg):
        self.left_img = self._camera_imgmsg_to_cv2(img_msg)
        self.on_left_image_received(self.left_img)

    def _on_right_imagemsg_received(self, img_msg):
        self.right_img = self._camera_imgmsg_to_cv2(img_msg)
        self.on_right_image_received(self.right_img)

    def _on_left_itbmsg_received(self, itb_msg):
        self.left_itb = itb_msg
        self.on_left_itb_received(self.left_itb)

    def _on_right_itbmsg_received(self, itb_msg):
        self.right_itb = itb_msg
        self.on_right_itb_received(self.right_itb)
    