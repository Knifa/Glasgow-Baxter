#!/usr/bin/env python

import cv2
import numpy as np

from geometry_msgs.msg import (
    Polygon, 
    Pose2D, 
    Point32
)

from glasgow_baxter.msg import (
    DetectedSquare, 
    TrackedSquare as TrackedSquareMsg
)

import itertools
import random

####################################################################################################

class Square(object):
    def __init__(self, contour, hue=None, tracking_id=None):
        self.contour = np.int0(contour.reshape((4, 2)))
        self.box = np.int0(cv2.cv.BoxPoints(cv2.minAreaRect(self.contour)))

        self.moments = cv2.moments(np.float32([self.box]))
        self.center = np.array([
            self.moments['m10'] / self.moments['m00'], 
            self.moments['m01'] / self.moments['m00']])

        self.hue = hue

    ############################################################################

    @staticmethod
    def from_msg(msg):
        contour = []
        for p in msg.screen_contour.points:
            contour.append([[p.x, p.y]])

        return Square(np.array(contour, dtype=np.int0), msg.hue)

    def to_msg(self):
        s_msg = DetectedSquare()
        s_msg.screen_pose.x = self.center[0]
        s_msg.screen_pose.y = self.center[1]
        s_msg.hue = self.hue

        for p in self.contour:
            s_msg.screen_contour.points.append(Point32(x=p[0], y=p[1], z=0))

        return s_msg

####################################################################################################

class TrackedSquare(Square):
    TRACKING_COLOURS = list(itertools.product([0, 128, 255], repeat=3))[1:]

    def __init__(self, detected_square):
        self.contour = detected_square.contour
        self.box = detected_square.box
        self.moments = detected_square.moments
        self.center = detected_square.center
        self.hue = detected_square.hue

        self.tracking_detected = False
        self.tracking_colour = random.choice(self.TRACKING_COLOURS)
        self.tracking_id = id(self)

    def to_msg(self):
        ts_msg = TrackedSquareMsg()
        ts_msg.detected_square = super(TrackedSquare, self).to_msg()

        ts_msg.tracking_detected = self.tracking_detected
        ts_msg.tracking_colour = list(self.tracking_colour)
        ts_msg.tracking_id = self.tracking_id

        return ts_msg

    @staticmethod
    def from_msg(msg):
        if type(msg) is DetectedSquare:
            return TrackedSquare(Square.from_msg(msg))
        elif type(msg) is TrackedSquareMsg:
            s = TrackedSquare(Square.from_msg(msg.detected_square))

            s.tracking_detected = msg.tracking_detected
            s.tracking_id = msg.tracking_id
            s.tracking_colour = tuple(map(lambda x: int(x), msg.tracking_colour))

            return s