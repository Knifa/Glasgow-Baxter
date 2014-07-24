#!/usr/bin/env python

from itertools import combinations

import cv2

import numpy as np
from scipy.spatial.distance import euclidean

import rospy
from geometry_msgs.msg import Polygon, Pose2D, Point32
from glasgow_baxter.msg import DetectedSquare, DetectedSquares

from glasgow_baxter_helpers import BaxterNode
from square import Square

####################################################################################################

class PerceptionNode(BaxterNode):
    SHAPE_STD_LIMIT = 3.0
    SHAPE_AREA_PERCENT_LIMIT = 0.2
    SHAPE_MIN_DISTANCE = 32
    SHAPE_ARC = 0.025
    RES_PERCENT = 0.66
    CANNY_DILATE = 3

    ############################################################################

    def __init__(self):
        super(PerceptionNode, self).__init__()

        self._fd = cv2.ORB()
        self._last_img = None
        self._last_kp = None

        self._squares_pub = rospy.Publisher(
            '/squares', 
            DetectedSquares,
            tcp_nodelay=True)

    ############################################################################

    def start(self):
        super(PerceptionNode, self).start(spin=True)

    ############################################################################

    def on_right_image_received(self, img):
        img = cv2.resize(img, 
            (int(img.shape[1] * self.RES_PERCENT), int(img.shape[0] * self.RES_PERCENT)))
        img = cv2.bilateralFilter(img, 3, 15, 15)

        squares = self._find_squares(img)
        self._publish_squares(squares)

    ############################################################################

    def _find_squares(self, img):
        contours = self._find_contours(img)
        squares = self._find_squares_from_contours(contours)
        squares = self._filter_squares(squares)
        squares = self._find_square_hue(squares, img)

        return squares

    def _find_square_hue(self, squares, img):
        for s in squares:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int0(s.box * self.RES_PERCENT), (255))

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv_channels = cv2.split(img_hsv)

            s.hue = np.median(img_hsv_channels[0][mask == 255])

        return squares

    def _find_squares_from_contours(self, contours):
        squares = []
        for c in contours:
            # Drop anything that doesn't have four corners (obviously.)
            if not len(c) == 4:
                continue

            # Find the distances between each side
            sides = []
            for i in range(len(c)):
                sides.append((c[i][0], c[(i + 1) % len(c)][0]))
            d = [euclidean(s[0], s[1]) for s in sides]

            # Drop any contours that doesn't have about the same size sides.
            d_std = np.std(d)
            if not (d_std > 0.0 and d_std < self.SHAPE_STD_LIMIT):
                continue

            # Drop any contours where the area isn't about the same value as a square with sides
            # equaling to the average side length.
            area = cv2.contourArea(c) 
            square_area = np.mean(d) ** 2
            area_dif = abs(square_area - area)
            if not area_dif <= square_area * self.SHAPE_AREA_PERCENT_LIMIT:
                continue

            squares.append(Square(c / self.RES_PERCENT))

        return squares

    def _filter_squares(self, squares):
        areas = [s.moments['m00'] for s in squares]
        std = np.std(areas)
        mean = np.mean(areas)

        # Remove any squares that are kind of outside the general area range.
        outlier_squares = list(filter(lambda s: abs(mean - s.moments['m00']) >= std * 2, squares))
        for s in outlier_squares:
            squares.remove(s)

        # Remove any squares that are way too close.
        for s1, s2 in combinations(squares, 2):
            if s1 in squares and euclidean(s1.center, s2.center) < self.SHAPE_MIN_DISTANCE:
                squares.remove(s1)

        return squares

    def _publish_squares(self, squares):
        s_msgs = DetectedSquares()
        for s in squares:
           s_msgs.squares.append(s.to_msg())

        self._squares_pub.publish(s_msgs)

    ############################################################################

    def _find_contours(self, img):
        # Detect contours in the image.
        canny_img = cv2.Canny(img, 50, 150)
        canny_img = cv2.dilate(canny_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
            (self.CANNY_DILATE, self.CANNY_DILATE)))
        (contours, _) = cv2.findContours(canny_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Smooth out the contours a little.
        approx_contours = []
        for c in contours:
            arc = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.SHAPE_ARC * arc, True)
            approx_contours.append(approx)

        return approx_contours

    def _optical_flow(self, in_img):
        img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)

        kp = self._fd.detect(img)
        kp = np.array([[k.pt] for k in kp2], np.float32)

        if self._last_kp is None:
            self._last_kp = kp
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self._last_img, img, self._last_kp)

            good_new = p1[st==1]
            good_old = self._last_kp[st==1]

            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()

                cv2.line(img, (a,b),(c,d), (255, 255, 255), 2)
                cv2.circle(img, (a,b), 5, (255, 255, 255), -1)
            
            self._last_kp = good_new.reshape(-1, 1, 2)         
        
        self._last_img = gray

####################################################################################################

def main():
    rospy.init_node('perception', anonymous=True)

    node = PerceptionNode()
    node.start()

if __name__ == '__main__':
    main()