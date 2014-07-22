#!/usr/bin/env python
import random

import numpy as np
import math

import cv2
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image, Range

from baxter_ikhelper import IKHelper
from baxter_core_msgs.msg import ITBState
from baxter_interface import (
    RobotEnable,
    AnalogIO, 
    DigitalIO,
    Gripper
)

from block_tracker import BlockTracker
from state_machine import StateMachine, State

####################################################################################################

class BlockStackerNode(object):
    RATE = 250 

    ############################################################################

    def __init__(self):
        self._rs = RobotEnable()
        self._cvbr = CvBridge()
        self._sm = RobotStateMachine(self)

        self.ik = IKHelper()
        self.ik.set_right(0.5, 0.0, 0.0, wait=True)

        self.left_img = None
        self.right_img = None
        self._left_camera_sub = rospy.Subscriber(
            '/cameras/left_hand_camera/image_rect_avg', 
            Image,
            self.on_left_imagemsg_received)
        self._right_camera_sub = rospy.Subscriber(
            '/cameras/right_hand_camera/image_rect_avg', 
            Image,
            self.on_right_imagemsg_received)
        self._display_pub = rospy.Publisher(
            '/robot/xdisplay', 
            Image, 
            tcp_nodelay=True,
            latch=True)

        self.range = None
        self._range_sub = rospy.Subscriber(
            '/robot/range/right_hand_range/state',
            Range,
            self.on_rangemsg_received)

        self.itb = None
        self._itb_sub = rospy.Subscriber(
            '/robot/itb/right_itb/state',
            ITBState,
            self.on_itbmsg_received)

        self.gripper = Gripper('right')
        self.gripper.calibrate()
        self.gripper.close(block=True)

    ############################################################################

    def start(self):
        self._rs.enable()
        self._sm.start()

        rate = rospy.Rate(BlockStackerNode.RATE)
        while not rospy.is_shutdown():
            self._sm.run_step()
            rate.sleep()

    ############################################################################

    def on_left_imagemsg_received(self, img_msg):
        img = self._cvbr.imgmsg_to_cv2(img_msg, 'bgr8')
        img = cv2.resize(img, (640, 400))

        self.left_img = img.copy()
        self._sm.on_left_image_received(img)

    def on_right_imagemsg_received(self, img_msg):
        img = self._cvbr.imgmsg_to_cv2(img_msg, 'bgr8')
        img = cv2.resize(img, (640, 400))

        self.right_img = img.copy()
        self._sm.on_right_image_received(img)

    def on_rangemsg_received(self, range_msg):
        self.range = range_msg.range

    def on_itbmsg_received(self, itb_msg):
        self.itb = itb_msg

    def display_image(self, img):
        img = cv2.resize(img, (1024, 600))

        print img.shape
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_msg = self._cvbr.cv2_to_imgmsg(img, 'bgr8')
        self._display_pub.publish(img_msg)
    
####################################################################################################

class RobotStateMachine(StateMachine):
    def __init__(self, node):
        super(RobotStateMachine, self).__init__(node)

        RobotStateMachine.calibrate = CalibrateState(node)

        RobotStateMachine.detect_estimate = DetectEstimateState(node)
        RobotStateMachine.detect_center = DetectCenterState(node)

        RobotStateMachine.pick = PickState(node)
        RobotStateMachine.place = PlaceState(node)
        
        self._current_state = RobotStateMachine.calibrate

####################################################################################################
    
class CalibrateState(State):
    def __init__(self, node):
        super(CalibrateState, self).__init__(node)

        self.table_z = None
        self.table_raw_z = None

        self.block_z = None
        self.block_raw_z = None
        self.block_height = None

        self.position_x_dif = None
        self.position_y_dif = None

    ############################################################################

    def enter(self):
        self.table_z = rospy.get_param('~table_z')
        self.table_raw_z = rospy.get_param('~table_raw_z')

        self.block_z = rospy.get_param('~block_z')
        self.block_raw_z = rospy.get_param('~block_raw_z')
        self.block_height = rospy.get_param('~block_height')

        self.position_x_dif = rospy.get_param('~position_x_dif')
        self.position_y_dif = rospy.get_param('~position_y_dif')

    def next(self):
        return RobotStateMachine.detect_estimate

    ############################################################################

    def calc_pos(self, screen_x, screen_y):
        cur_pos = self._node.ik.get_right()
        return (
            cur_pos.x + (self.position_x_dif * screen_x), 
            cur_pos.y + (self.position_y_dif * -screen_y))

####################################################################################################

class DetectEstimateState(State):
    def __init__(self, node):
        super(DetectEstimateState, self).__init__(node)

        self.bt = BlockTracker()

        self._ready = False
        self._block_centered = False
        self._got_image = False

        self._z = None

    ############################################################################

    def enter(self):
        self._ready = False
        self._block_centered = False
        self._got_image = False

        self._z = RobotStateMachine.calibrate.table_z + \
            (RobotStateMachine.calibrate.block_height * 2)

        self._node.ik.set_right(0.5, 0.0, self._z, wait=True)
        self._ready = True

    def next(self):
        if not self._block_centered:
            return self
        else:
            return RobotStateMachine.detect_center

    ############################################################################

    def on_right_image_received(self, img):
        if self._got_image or not self._ready:
            return

        self.bt.on_image_received(img)
        self._got_image = True

        if not self.bt.display_img is None:
            self._node.display_image(self.bt.display_img)

        self._block = random.choice(self.bt.blocks)
        self._block_pos = RobotStateMachine.calibrate.calc_pos(
            self._block.rel_pos[0], self._block.rel_pos[1])

        self._node.ik.set_right(self._block_pos[0], self._block_pos[1], self._z, wait=True)
        self._block_centered = True

####################################################################################################

class DetectCenterState(State):
    _MOVE_RATE = (1.0 / BlockStackerNode.RATE) * 0.1
    _THRESHOLD = 0.05

    ############################################################################

    def __init__(self, node):
        super(DetectCenterState, self).__init__(node)

        self._x = 0
        self._y = 0
        self._z = 0

        self._centered = False

        self.bt = BlockTracker()

    ############################################################################

    def enter(self):
        self._centered = False

        self._x = RobotStateMachine.detect_estimate._block_pos[0]
        self._y = RobotStateMachine.detect_estimate._block_pos[1]
        self._z = RobotStateMachine.calibrate.table_z + \
            (RobotStateMachine.calibrate.block_height * 2)

    def run_step(self):
        if not self._centered:
            self._do_centering()

    def next(self):
        if not self._centered:
            return self
        else:
            return RobotStateMachine.pick

    ############################################################################

    def _do_centering(self):
        if not len(self.bt.blocks) > 0:
            return

        target = (0.1, -0.5)
        b = self.bt.blocks[0]
        dif = (target[0] - b.rel_pos[0], target[1] - b.rel_pos[1])
        
        if abs(dif[0]) > self._THRESHOLD or abs(dif[1]) > self._THRESHOLD:
            if abs(dif[0]) > self._THRESHOLD:
                self._x += math.copysign(self._MOVE_RATE, -dif[0]) * np.clip(abs(dif[0]), 0.1, 1.0)
            if abs(dif[1]) > self._THRESHOLD:
                self._y += math.copysign(self._MOVE_RATE, dif[1]) * np.clip(abs(dif[1]), 0.1, 1.0)
        else:
            self._centered = True

        self._node.ik.set_right(self._x, self._y, self._z)

    ############################################################################

    def on_right_image_received(self, img):
        self.bt.on_image_received(img)

        if not self.bt.display_img is None:
            self._node.display_image(self.bt.display_img) 

####################################################################################################

class PickState(State):
    _Z_RATE = (1.0 / BlockStackerNode.RATE) * 0.1

    ############################################################################

    def __init__(self, node):
        super(PickState, self).__init__(node)

        self._z = None
        self._start_z = None
        self._target_z = None

        self._picked = False

    ############################################################################

    def enter(self):
        rospy.sleep(1.0)

        cur_pos = self._node.ik.get_right()
        self._x = cur_pos[0]
        self._y = cur_pos[1]

        self._start_z = RobotStateMachine.calibrate.table_z + \
            RobotStateMachine.calibrate.block_height
        self._target_z = RobotStateMachine.calibrate.table_z

        self._z = self._start_z

        self._picked = False

        self._node.ik.set_right(self._x, self._y, self._z, wait=True)
        self._node.gripper.open(block=True)
        rospy.sleep(1.0)

    def run_step(self):
        if not self._picked:
            if self._z >= self._target_z:
                self._z -= PickState._Z_RATE
                self._node.ik.set_right(self._x, self._y, self._z)
            else:
                rospy.sleep(1.0)
                self._node.gripper.close(block=True)
                rospy.sleep(1.0)
                self._node.ik.set_right(self._x, self._y, self._start_z, wait=True)

                self._picked = True

    def next(self):
        if self._picked:
            return RobotStateMachine.place
        else:
            return RobotStateMachine.pick

####################################################################################################

class PlaceState(State):
    _Z_RATE = (1.0 / BlockStackerNode.RATE) * 0.1
    _ACCEPT_FORCE = -10

    ############################################################################

    def __init__(self, node):
        super(PlaceState, self).__init__(node)

        self._placed = False
        self._h = 0

    ############################################################################

    def enter(self):
        self._place_point = [0.5, -0.25, RobotStateMachine.calibrate.table_z]

        self._start_z = RobotStateMachine.calibrate.table_z + (RobotStateMachine.calibrate.block_height * (self._h + 1))
        self._target_z = RobotStateMachine.calibrate.table_z + (self._h * RobotStateMachine.calibrate.block_height)
        self._z = self._start_z

        self._placed = False

        self._node.ik.set_right(
            self._place_point[0], 
            self._place_point[1], 
            self._start_z, 
            wait=True)

    def run_step(self):
        if not self._placed:

            f = self._node.ik.get_right_force().z
            v = self._node.ik.get_right_velocity().z

            if abs(f) > 10 and abs(v) <= 0.005:
                print f, v

            if self._node.ik.get_right_force().z <= PlaceState._ACCEPT_FORCE:
                rospy.sleep(1.0) 
                self._node.gripper.open(block=True)
                self._node.ik.set_right(
                    self._place_point[0], 
                    self._place_point[1], 
                    self._start_z + (RobotStateMachine.calibrate.block_height * 2), 
                    wait=True)

                self._h += 1
                self._placed = True
            else:
                self._z -= PlaceState._Z_RATE
                self._node.ik.set_right(self._place_point[0], self._place_point[1], self._z)                

    def next(self):
        if not self._placed:
            return RobotStateMachine.place
        else:
            return RobotStateMachine.detect_estimate

####################################################################################################

def main():
    rospy.init_node('block_stacker', anonymous=True)

    node = BlockStackerNode()
    node.start()

if __name__ == '__main__':
    main()