#!/usr/bin/env python
import numpy as np
import math
import yaml
import cv2
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image
from baxter_ikhelper import IKHelper
from baxter_core_msgs.msg import ITBState
from baxter_interface import (
    RobotEnable,
    DigitalIO,
    Gripper
)

from block_tracker import BlockTracker
from state_machine import StateMachine, State

####################################################################################################

class BlockStackerCalibrateNode(object):
    RATE = 250 

    ############################################################################

    def __init__(self):
        self._rs = RobotEnable()
        self._cvbr = CvBridge()
        self._sm = RobotStateMachine(self)

        self.ik = IKHelper()
        self.ik.set_right(0.5, 0.0, 0.0, wait=True)

        self.gripper = Gripper('right')
        self.gripper.calibrate()
        self.gripper.close(block=True)

        self.right_img = None
        self._right_camera_sub = rospy.Subscriber(
            '/cameras/right_hand_camera/image_rect_avg', 
            Image,
            self.on_right_imagemsg_received)

        self.itb = None
        self._itb_sub = rospy.Subscriber(
            '/robot/itb/right_itb/state',
            ITBState,
            self.on_itbmsg_received)

    	self._display_pub = rospy.Publisher(
            '/robot/xdisplay', 
            Image, 
            tcp_nodelay=True,
            latch=True)

    ############################################################################

    def start(self):
        self._rs.enable()
        self._sm.start()

        rate = rospy.Rate(self.RATE)
        while not rospy.is_shutdown():
            self._sm.run_step()
            rate.sleep()

    ############################################################################

    def on_right_imagemsg_received(self, img_msg):
        img = self._cvbr.imgmsg_to_cv2(img_msg, 'bgr8')
        img = cv2.resize(img, (640, 400))

        self.right_img = img.copy()
        self._sm.on_right_image_received(img)

    def on_itbmsg_received(self, itb_msg):
        self.itb = itb_msg

    ############################################################################

    def display_image(self, img):
        img = cv2.resize(img, (1024, 600))
        img_msg = self._cvbr.cv2_to_imgmsg(img, 'bgr8')
        self._display_pub.publish(img_msg)

####################################################################################################

class RobotStateMachine(StateMachine):
    def __init__(self, node):
    	super(RobotStateMachine, self).__init__(node)

        RobotStateMachine.calibrate_table_height = CalibrateTableHeightState(node)
        RobotStateMachine.calibrate_block_height = CalibrateBlockHeightState(node)
        RobotStateMachine.calibrate_position_estimation = CalibratePositionEstimationState(node)
        RobotStateMachine.output_yaml = OutputYamlState(node)

        self._current_state = RobotStateMachine.calibrate_table_height

####################################################################################################
    
class CalibrateTableHeightState(State):
    _Z_START = 0.0
    _Z_RATE = (1.0 / BlockStackerCalibrateNode.RATE) * 0.25
    _Z_BACKSTEP = 0.02
    _ACCEPT_FORCE = -10.0
    _AVG_NUM = 1

    ############################################################################

    def __init__(self, node):
        super(CalibrateTableHeightState, self).__init__(node)

        self.table_z = None
        self.table_raw_z = None

        self._calibrated = False
        self._calibration_triggered = False

        self._z = self._Z_START
        self._avg_total = 0
        self._avg_count = 0

    ############################################################################

    def enter(self):
        while self._node.itb is None:
            pass

        print('searching for table')
        self._reset()

    def _reset(self):
        self._z = self._Z_START
        self._node.ik.set_right(0.5, 0.0, self._z, wait=True)

    def run_step(self):
    	if not self._calibration_triggered:
            self._calibration_triggered = self._node.itb.buttons[0]
            if self._calibration_triggered:
                rospy.sleep(1.0)
            return

        if self._node.ik.get_right_force().z <= self._ACCEPT_FORCE: 
            self._avg_total += self._node.ik.get_right().z
            self._avg_count += 1

            if self._avg_count >= self._AVG_NUM:
                self.table_raw_z = self._avg_total / self._avg_count       
                self.table_z = self.table_raw_z + self._Z_BACKSTEP
                self._calibrated = True

                print('found table at:')
                print('\tz: {0}m'.format(self.table_z))
                print('\traw z: {0}m'.format(self.table_raw_z))
                print('')

            self._reset()
        else:
            self._z -= self._Z_RATE
            self._node.ik.set_right(0.5, 0.0, self._z)

    def next(self):
        if not self._calibrated:
            return self
        else:
            return RobotStateMachine.calibrate_block_height

####################################################################################################

class CalibrateBlockHeightState(State):
    _Z_START = 0.0
    _Z_RATE = (1.0 / BlockStackerCalibrateNode.RATE) * 0.25
    _Z_BACKSTEP = 0.02
    _ACCEPT_FORCE = -10.0
    _AVG_NUM = 1

    ############################################################################

    def __init__(self, node):
        super(CalibrateBlockHeightState, self).__init__(node)

        self.block_z = None
        self.block_raw_z = None
        self.block_height = None

        self._calibrated = False
        self._calibration_triggered = False

        self._z = self._Z_START
        self._avg_total = 0
        self._avg_count = 0
        
    ############################################################################

    def enter(self):
        while self._node.itb is None:
            pass

        print('searching for block')
        self._reset()
        
    def _reset(self):
        self._z = self._Z_START
        self._node.ik.set_right(0.5, 0.0, self._z, wait=True)

    def run_step(self):
        if not self._calibration_triggered:
            self._calibration_triggered = self._node.itb.buttons[0]
            if self._calibration_triggered:
                rospy.sleep(1.0)
            return

        if self._node.ik.get_right_force().z <= self._ACCEPT_FORCE: 
            self._avg_total += self._node.ik.get_right().z
            self._avg_count += 1

            if self._avg_count >= self._AVG_NUM:
                self.block_raw_z = self._avg_total / self._avg_count  
                self.block_z = self.block_raw_z + self._Z_BACKSTEP
                self.block_height = abs(
                    RobotStateMachine.calibrate_table_height.table_raw_z - self.block_raw_z)

                self._calibrated = True

                print('found block at:')
                print('\tz: {0}m'.format(self.block_z))
                print('\traw z: {0}m'.format(self.block_raw_z))
                print('\theight: {0}m'.format(self.block_height))
                print('')

            self._reset()
        else:
            self._z -= self._Z_RATE
            self._node.ik.set_right(0.5, 0.0, self._z)

    def next(self):
        if not self._calibrated:
            return self
        else:
            return RobotStateMachine.calibrate_position_estimation

####################################################################################################

class CalibratePositionEstimationState(State):
    _MOVE_RATE = (1.0 / BlockStackerCalibrateNode.RATE) * 0.1
    _THRESHOLD = 0.05

    _TARGET_DISTANCE = 0.5
    _TARGETS = {
        'center': (0, 0),
        'left': (-_TARGET_DISTANCE, 0),
        'right': (_TARGET_DISTANCE, 0),
        'up': (0, -_TARGET_DISTANCE),
        'down': (0, _TARGET_DISTANCE),
    }

    ############################################################################

    def __init__(self, node):
        super(CalibratePositionEstimationState, self).__init__(node)

        self.bt = BlockTracker()

        self._x = 0.5
        self._y = 0
        self._z = 0

        self._target_keys = self._TARGETS.keys()
        self._target = self._target_keys[0]  
        self._target_pos = dict([(x, None) for x in self._target_keys])

        self._target_pos_calibrated = False
        self._target_pos_computed = False

        self.x_dif = None
        self.y_dif = None        

    ############################################################################

    def enter(self):
        self._z = RobotStateMachine.calibrate_table_height.table_z + \
            (RobotStateMachine.calibrate_block_height.block_height * 2)

        print('estimating screen/arm positions...')

    def run_step(self):
        if not self._target_pos_calibrated:
            self._do_target_pos_calibration()
        elif not self._target_pos_computed:
            self._do_target_pos_computation()

    def next(self):
        if not self._target_pos_calibrated:
            return self
        elif not self._target_pos_computed:
        	return self
        else:
            return RobotStateMachine.output_yaml

    ############################################################################

    def _do_target_pos_calibration(self):
        if not len(self.bt.blocks) > 0:
            return

        target = self._TARGETS[self._target]
        b = self.bt.blocks[0]
        dif = (target[0] - b.rel_pos[0], target[1] - b.rel_pos[1])
        
        if abs(dif[0]) > self._THRESHOLD or abs(dif[1]) > self._THRESHOLD:
            if abs(dif[0]) > self._THRESHOLD:
                self._x += math.copysign(self._MOVE_RATE, -dif[0]) * np.clip(abs(dif[0]), 0.1, 1.0)
            if abs(dif[1]) > self._THRESHOLD:
                self._y += math.copysign(self._MOVE_RATE, dif[1]) * np.clip(abs(dif[1]), 0.1, 1.0)
        else:
            self._target_pos[self._target] = [self._x, self._y]

            i = self._target_keys.index(self._target) + 1
            if not i >= len(self._target_keys):
                self._target = self._target_keys[i]
            else:
                self._target_pos_calibrated = True
                print('got all positions')

        self._node.ik.set_right(self._x, self._y, self._z)

    def _do_target_pos_computation(self):
    	print('combobulating...')

        l_dif = abs(self._target_pos['left'][0] - self._target_pos['center'][0])
        r_dif = abs(self._target_pos['right'][0] - self._target_pos['center'][0])
        u_dif = abs(self._target_pos['up'][1] - self._target_pos['center'][1])
        d_dif = abs(self._target_pos['down'][1] - self._target_pos['center'][1])

        self.x_dif = (l_dif + r_dif) / 2 / self._TARGET_DISTANCE
        self.y_dif = (u_dif + d_dif) / 2 / self._TARGET_DISTANCE

        self._target_pos_computed = True

    ############################################################################

    def on_right_image_received(self, img):
        self.bt.on_image_received(img)

        if not self.bt.display_img is None:
            self._node.display_image(self.bt.display_img) 

####################################################################################################

class OutputYamlState(State):
	def __init__(self, node):
		super(OutputYamlState, self).__init__(node)

	def enter(self):
		print('ok paste this into a cool file:\n')
		print yaml.dump({
			'table_z': RobotStateMachine.calibrate_table_height.table_z,
			'table_raw_z': RobotStateMachine.calibrate_table_height.table_raw_z,

			'block_z': RobotStateMachine.calibrate_block_height.block_z,
			'block_raw_z': RobotStateMachine.calibrate_block_height.block_raw_z,
			'block_height': RobotStateMachine.calibrate_block_height.block_height,

			'position_x_dif': float(RobotStateMachine.calibrate_position_estimation.x_dif),
			'position_y_dif': float(RobotStateMachine.calibrate_position_estimation.y_dif)
		})

	def next(self):
		return None

####################################################################################################

def main():
    rospy.init_node('block_stacker_calibrate', anonymous=True)

    node = BlockStackerCalibrateNode()
    node.start()

if __name__ == '__main__':
    main()