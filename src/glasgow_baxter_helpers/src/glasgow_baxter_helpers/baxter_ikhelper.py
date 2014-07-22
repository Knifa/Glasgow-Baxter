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

_IKSVC_LEFT_URI = 'ExternalTools/left/PositionKinematicsNode/IKService'
_IKSVC_RIGHT_URI = 'ExternalTools/right/PositionKinematicsNode/IKService'

################################################################################

class IKHelper(object):
    """An abstraction layer for using Baxter's built in IK service."""

    ############################################################################

    def __init__(self):
        self._left_arm = Limb('left')
        self._left_arm.set_joint_position_speed(0.3)
        self._right_arm = Limb('right')
        self._right_arm.set_joint_position_speed(0.3)

        self._left_iksvc = rospy.ServiceProxy(
            _IKSVC_LEFT_URI,
            SolvePositionIK)

        self._right_iksvc = rospy.ServiceProxy(
            _IKSVC_RIGHT_URI,
            SolvePositionIK)

        self._joint_update_pub = rospy.Publisher(
            '/robot/joint_state_publish_rate', 
            UInt16)
        self._joint_update_pub.publish(250)

    ############################################################################

    def reset(self):
        """Reset both arms to their neutral positions."""   
    	self._left_arm.move_to_neutral()
        self._right_arm.move_to_neutral()

    def set_left(self, pos, rot=(0, math.pi, math.pi *0.5), wait=False):
        """Set the endpoint of the left arm to the supplied coordinates.

        Arguments:
            pos -- Position in space in (x, y, z) format.
            rot -- Rotation in space in (r, p, y) format. (defaults to pointing
                downwards.)

        Keyword arguments:
            wait -- If True, method will block until in position. (default 
                False)
        """
        self._set_arm(self._left_iksvc, self._left_arm, pos, rot, wait)

    def set_left(self, pos, rot=(0, math.pi, math.pi *0.5), wait=False):
        """Set the endpoint of the right arm to the supplied coordinates.

        Arguments:
            pos -- Position in space in (x, y, z) format.
            rot -- Rotation in space in (r, p, y) format. (defaults to pointing
                downwards.)

        Keyword arguments:
            wait -- If True, method will block until in position. (default 
                False)
        """
        self._set_arm(self._right_iksvc, self._right_arm, pos, rot, wait)

    def get_left(self):
        """Return the current endpoint pose of the left arm."""
        return self._left_arm.endpoint_pose()['position']
    
    def get_right(self):
        """Return the current endpoint pose of the left arm."""
        return self._right_arm.endpoint_pose()['position']

    def get_left_velocity(self):
        """Return the current endpoint velocity of the left arm."""
        return self._left_arm.endpoint_velocity()['linear']

    def get_right_velocity(self):
        """Return the current endpoint velocity of the right arm."""
        return self._right_arm.endpoint_velocity()['linear']

    def get_left_force(self):
        """Return the current endpoint force on the left arm."""
        return self._left_arm.endpoint_effort()['force']

    def get_right_force(self):
        """Return the current endpoint force on the right arm."""
        return self._right_arm.endpoint_effort()['force']

    ############################################################################

    def _set_arm(self, iksvc, limb, pos, rot, wait):
        resp = self._get_ik(iksvc, pos, rot)
        positions = resp[0]
        isValid = resp[1]
        if not isValid:
            print('invalid: {0} {1} {2}'.format(x, y, z))

        if not wait:
            limb.set_joint_positions(positions)
        else:
            limb.move_to_joint_positions(positions)

    def _get_ik(self, iksvc, pos, rot):
        q = quaternion_from_euler(rot[0], rot[1], rot[2])

        pose = PoseStamped(
            header=Header(stamp=rospy.Time.now(), frame_id='base'),
            pose=Pose(
                position=Point(
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
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