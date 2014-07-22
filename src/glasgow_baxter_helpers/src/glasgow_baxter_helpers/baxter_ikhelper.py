import math
import rospy

from tf.transformations import quaternion_from_euler
from std_msgs.msg import Header, UInt16
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from baxter_interface import RobotEnable, CameraController, Limb
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

################################################################################

IKSVC_LEFT_URI = 'ExternalTools/left/PositionKinematicsNode/IKService'
IKSVC_RIGHT_URI = 'ExternalTools/right/PositionKinematicsNode/IKService'

################################################################################

class IKHelper(object):
    def __init__(self):
        self._left_arm = Limb('left')
        self._left_arm.set_joint_position_speed(0.3)
        self._right_arm = Limb('right')
        self._right_arm.set_joint_position_speed(0.3)

        self._left_iksvc = rospy.ServiceProxy(
            IKSVC_LEFT_URI,
            SolvePositionIK)

        self._right_iksvc = rospy.ServiceProxy(
            IKSVC_RIGHT_URI,
            SolvePositionIK)

        self._joint_update_pub = rospy.Publisher('/robot/joint_state_publish_rate', UInt16)
        self._joint_update_pub.publish(250)

    def reset(self):
    	self._left_arm.move_to_neutral()
        self._right_arm.move_to_neutral()

    def set_left(self, x, y, z, wait=False):
        self._set_arm(self._left_iksvc, self._left_arm, x, y, z, wait)

    def set_right(self, x, y, z, wait=False):
        self._set_arm(self._right_iksvc, self._right_arm, x, y, z, wait)

    def get_left(self):
        return self._left_arm.endpoint_pose()['position']
    
    def get_right(self):
        return self._right_arm.endpoint_pose()['position']

    def get_left_velocity(self):
        return self._left_arm.endpoint_velocity()['linear']

    def get_right_velocity(self):
        return self._right_arm.endpoint_velocity()['linear']

    def get_left_force(self):
        return self._left_arm.endpoint_effort()['force']

    def get_right_force(self):
        return self._right_arm.endpoint_effort()['force']

    ############################################################################

    def _set_arm(self, iksvc, limb, x, y, z, wait):
        resp = self._get_ik(iksvc, x, y, z)
        positions = resp[0]
        isValid = resp[1]
        if not isValid:
            print('invalid: {0} {1} {2}'.format(x, y, z))

        if not wait:
            limb.set_joint_positions(positions)
        else:
            limb.move_to_joint_positions(positions)

    def _get_ik(self, iksvc, x, y, z):
        q = quaternion_from_euler(math.pi * 0, math.pi * 1, math.pi * 0.5)

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

        iksvc.wait_for_service(5.0)
        resp = iksvc(ikreq)

        positions = dict(zip(resp.joints[0].name, resp.joints[0].position))
        return (positions, resp.isValid[0])