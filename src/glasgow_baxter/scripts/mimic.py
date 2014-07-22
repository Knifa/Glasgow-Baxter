#!/usr/bin/env python
import math
import numpy as np
import cv2
from cv_bridge import CvBridge

import rospy
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from std_msgs.msg import (
    Header, 
    UInt16
)
from sensor_msgs.msg import (
    Image, 
    JointState
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from baxter_interface import (
    RobotEnable, 
    CameraController, 
    Limb
)
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

####################################################################################################

cvbr = CvBridge()

####################################################################################################

class Mimic(object):
    IKSVC_LEFT_URI = 'ExternalTools/left/PositionKinematicsNode/IKService'
    IKSVC_RIGHT_URI = 'ExternalTools/right/PositionKinematicsNode/IKService'

    def __init__(self):
        self._tf = tf.TransformListener()
        self._rs = RobotEnable()

        self._left_arm = Limb('left')
        self._left_arm.set_joint_position_speed(0.5)
        self._right_arm = Limb('right')
        self._right_arm.set_joint_position_speed(0.5)

        self._left_arm_neutral = None
        self._right_arm_neutral = None

        self._left_iksvc = rospy.ServiceProxy(
            Mimic.IKSVC_LEFT_URI,
            SolvePositionIK)

        self._right_iksvc = rospy.ServiceProxy(
            Mimic.IKSVC_RIGHT_URI,
            SolvePositionIK)

        self._left_camera = CameraController('left_hand_camera')
        self._right_camera = CameraController('right_hand_camera')
        self._head_camera = CameraController('head_camera')

        self._left_camera.close()
        self._right_camera.close()
        self._head_camera.close()

        self._head_camera.resolution = CameraController.MODES[0]
        self._head_camera.open()

        self._head_camera_sub = rospy.Subscriber('/cameras/head_camera/image', Image,
            self._head_camera_sub_callback)
        self._xdisplay_pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)

        self._out_of_range = False

        self._ik_smooth = 4
        self._ik_rate = 200
        self._avg_window = 1000

        self._r_trans_prev = []
        self._l_trans_prev = []

        self._r_ik_prev = []
        self._l_ik_prev = []

        self._joint_update_pub = rospy.Publisher('/robot/joint_state_publish_rate', UInt16)
        self._joint_update_pub.publish(self._ik_rate)

        self._mimic_timer = None
    
    ############################################################################

    def start(self):
        self._rs.enable()
        self._reset()

        self._mimic_timer = rospy.Timer(rospy.Duration(1.0 / self._ik_rate), self._mimic_callback)
        rate = rospy.Rate(self._avg_window)

        while not rospy.is_shutdown():
            try:
                (r_trans,r_rot) = self._tf.lookupTransform(
                    '/skeleton/user_1/right_hand', 
                    '/skeleton/user_1/torso', 
                    rospy.Time(0))

                (l_trans,l_rot) = self._tf.lookupTransform(
                    '/skeleton/user_1/left_hand', 
                    '/skeleton/user_1/torso', 
                    rospy.Time(0))

                self._l_trans_prev.append(l_trans)
                if (len(self._l_trans_prev) > self._avg_window):
                    self._l_trans_prev.pop(0)
                    
                self._r_trans_prev.append(r_trans)
                if (len(self._r_trans_prev) > self._avg_window):
                    self._r_trans_prev.pop(0)
                    
            except:
                self._out_of_range = True
                rate.sleep()
                continue
            
            rate.sleep()

    def _reset(self):
        if self._mimic_timer:
            self._mimic_timer.shutdown()

        self._left_arm.move_to_neutral()
        self._left_arm_neutral = self._left_arm.joint_angles()

        self._right_arm.move_to_neutral()
        self._right_arm_neutral = self._right_arm.joint_angles()

        print(self._left_arm_neutral)
        print(self._right_arm_neutral)

    def shutdown(self):
        self._reset()

    ############################################################################

    def _mimic_callback(self, event):
        try:
            l_trans = (
                sum(map(lambda x: x[0], self._l_trans_prev)) / self._avg_window,
                sum(map(lambda x: x[1], self._l_trans_prev)) / self._avg_window,
                sum(map(lambda x: x[2], self._l_trans_prev)) / self._avg_window)

            r_trans = (
                sum(map(lambda x: x[0], self._r_trans_prev)) / self._avg_window,
                sum(map(lambda x: x[1], self._r_trans_prev)) / self._avg_window,
                sum(map(lambda x: x[2], self._r_trans_prev)) / self._avg_window)

            rh_x = clamp(0.65 + (l_trans[2] / 1.5), 0.5, 0.9)
            rh_y = clamp(-l_trans[0], -0.8, 0)
            rh_z = clamp(-l_trans[1], -0.1, 0.8)

            lh_x = clamp(0.65 + (r_trans[2] / 1.5), 0.5, 0.9)
            lh_y = clamp(-r_trans[0], 0, 0.8)
            lh_z = clamp(-r_trans[1], -0.1, 0.8)

            self.set_left_coords(lh_x, lh_y, lh_z)
            self.set_right_coords(rh_x, rh_y, rh_z)
        except:
            return

    def _head_camera_sub_callback(self, data):
        orig_img = cvbr.imgmsg_to_cv2(data, 'rgb8')
        img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)

        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 100, 200)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.addWeighted(img, 0.75, orig_img, 0.25, 0)
        img = cv2.resize(img, (1024, 768))
        if self._out_of_range:
            cv2.putText(img, 'Out of range!', (50, 768-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cvbr.cv2_to_imgmsg(img)
        self._xdisplay_pub.publish(img)

    ############################################################################

    def set_left_coords(self, 
                        x, y, z, 
                        er=math.pi * -1, ep=math.pi * 0.5, ey=math.pi * -1):
        self._set_arm_coords(self._left_iksvc, self._left_arm, 
                             x, y, z, 
                             er, ep, ey,
                             self._get_neutral_joint_state(self._left_arm_neutral),
                             self._l_ik_prev)

    def set_right_coords(self, 
                         x, y, z,
                         er=math.pi * -1, ep=math.pi * 0.5, ey=math.pi * -1):
        self._set_arm_coords(self._right_iksvc, self._right_arm, 
                             x, y, z, 
                             er, ep, ey,
                             self._get_neutral_joint_state(self._right_arm_neutral),
                             self._r_ik_prev)

    ############################################################################

    def _set_arm_coords(self, iksvc, limb, x, y, z, er, ep, ey, njs, prev):
        resp = self._get_ik(iksvc, x, y, z, er, ep, ey, njs)
        positions = resp[0]
        isValid = resp[1]
        self._out_of_range = not isValid

        prev.append(positions)
        if (len(prev) > self._ik_smooth):
            prev.pop(0)
        
        smooth_positions = {}
        for joint_name in positions:
            smooth_positions[joint_name] = \
                sum(map(lambda x: x[joint_name], prev)) / self._ik_smooth

        limb.set_joint_positions(smooth_positions)

    def _get_ik(self, iksvc, x, y, z, er, ep, ey, njs):
        q = quaternion_from_euler(er, ep, ey)

        pose = PoseStamped(
            header=Header(stamp=rospy.Time.now(), frame_id='base'),
            pose=Pose(
                position=Point(
                    x=x,
                    y=y,
                    z=z,
                ),
                orientation=Quaternion(q[0], q[1], q[2], q[3])
            ),
        )
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(pose)
        ikreq.seed_angles.append(njs)

        iksvc.wait_for_service(1.0)
        resp = iksvc(ikreq)

        positions = dict(zip(resp.joints[0].name, resp.joints[0].position))
        return (positions, resp.isValid[0])

    def _get_neutral_joint_state(self, njs):
        js = JointState()
        js.header = Header(stamp=rospy.Time.now(), frame_id='base')

        for j in njs:
            js.name.append(j)
            js.position.append(njs[j])

        return js

####################################################################################################

def clamp(val, minval, maxval):
    if val < minval:
        return minval
    elif val > maxval:
        return maxval
    else:
        return val

####################################################################################################

def main():
    rospy.init_node('mimic', anonymous=True)

    mimic = Mimic()
    rospy.on_shutdown(mimic.shutdown)
    mimic.start()

####################################################################################################

if __name__ == '__main__':
    main()