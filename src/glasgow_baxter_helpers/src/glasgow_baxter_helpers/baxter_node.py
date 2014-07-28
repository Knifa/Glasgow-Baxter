import cv2
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image, Range

from baxter_core_msgs.msg import ITBState
from baxter_interface import (
    RobotEnable,
    AnalogIO, 
    DigitalIO,
    Gripper,
    Head
)

from baxter_ikhelper import IKHelper

################################################################################

class BaxterNode(object):
    """Helper class for creating Baxter nodes quickly, intended to be 
    sublclassed.

    Attributes:
        rs -- An instance of RobotEnable provided by baxter_interface.
        ik -- An instance of IKHelper.

        left_img -- Last recieved image from left camera, rectified and as an 
            OpenCV numpy array.
        right_img -- Last recieved image from right camera, rectified and as an 
            OpenCV numpy array.

        left_itb -- Last received state from left ITB (shoulder buttons).
        left_itb -- Last received state from right ITB (shoulder buttons).

        left_gripper -- Instance of Gripper from baxter_interface, for left
            hand.
        right_gripper -- Instance of Gripper from baxter_interface, for right
            hand.
    """

    ############################################################################

    def __init__(
            self, 
            camera_averaging=False):
        self._cvbr = CvBridge()

        self.rs = RobotEnable()
        self.ik = IKHelper()

        camera_topic = '/cameras/{0}_camera/image_rect'
        if camera_averaging:
            camera_topic += '_avg'

        self.left_img = None
        self._left_camera_sub = rospy.Subscriber(
            camera_topic.format('left_hand'), 
            Image,
            self._on_left_imagemsg_received)

        self.right_img = None
        self._right_camera_sub = rospy.Subscriber(
            camera_topic.format('right_hand'), 
            Image,
            self._on_right_imagemsg_received)

        self.head_img = None
        self._head_camera_sub = rospy.Subscriber(
            camera_topic.format('head'), 
            Image,
            self._on_head_imagemsg_received)

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

        self.head = Head()

        self._display_pub = rospy.Publisher(
            '/robot/xdisplay', 
            Image, 
            tcp_nodelay=True,
            latch=True)

    ############################################################################

    def start(self, spin=False, calibrate=False):
        """Start up the node and initalise Baxter.

        Keyword arguments:
            spin -- Enter a spin loop once initialised (default False).
            calibrate -- Calibrate the grippers (default False).
        """
        self.rs.enable()

        # Wait for initial topic messages to come in.
        while self.left_itb is None or \
                self.right_itb is None:
            rospy.sleep(100)

        # Calibrate both grippers.
        if calibrate:
            self.left_gripper.calibrate()
            self.right_gripper.calibrate()

        if spin:
            rospy.spin()

    def display_image(self, img):
        """Displays an image on the screen.

        Arguments:
            img -- A OpenCV numpy array to be displayed.
        """
        img = cv2.resize(img, (1024, 600))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_msg = self._cv2_to_display_imgmsg(img)
        self._display_pub.publish(img_msg)

    ############################################################################

    def on_left_image_received(self, img):
        """Called when a image is received from the left camera. Intended to be
        overridden.

        Arguments:
            img -- The rectified OpenCV numpy array from the camera.
        """
        pass

    def on_right_image_received(self, img):
        """Called when a image is received from the right camera. Intended to be
        overridden.

        Arguments:
            img -- The rectified OpenCV numpy array from the camera.
        """
        pass

    def on_head_image_received(self, img):
        """Called when a image is received from the head camera. Intended to be
        overridden.

        Arguments:
            img -- The rectified OpenCV numpy array from the camera.
        """
        pass

    def on_left_itb_received(self, itb):
        """Called when a left ITB state update is received. Intended to be
        overridden.

        Arguments:
            itb -- The new ITB state.
        """
        pass

    def on_right_itb_received(self, itb):
        """Called when a right ITB state update is received. Intended to be
        overridden.

        Arguments:
            itb -- The new ITB state.
        """
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

    def _on_head_imagemsg_received(self, img_msg):
        img = self._camera_imgmsg_to_cv2(img_msg)
        img = cv2.flip(img, 0)

        self.head_img = img
        self.on_head_image_received(self.head_img)

    def _on_left_itbmsg_received(self, itb_msg):
        self.left_itb = itb_msg
        self.on_left_itb_received(self.left_itb)

    def _on_right_itbmsg_received(self, itb_msg):
        self.right_itb = itb_msg
        self.on_right_itb_received(self.right_itb)
