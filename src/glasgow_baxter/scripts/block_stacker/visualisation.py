#!/usr/bin/env python

import numpy as np
import cv2

import rospy
from glasgow_baxter_helpers import BaxterNode
from glasgow_baxter.msg import DetectedSquares, TrackedSquares
from square import Square, TrackedSquare

from colorsys import hls_to_rgb

####################################################################################################

class VisualisationNode(BaxterNode):
    def __init__(self):
        super(VisualisationNode, self).__init__()

        self._squares_sub = rospy.Subscriber(
            '/tracked_squares', 
            TrackedSquares,
            self.on_squaremsg_received)

        self._squares = []

    ############################################################################

    def start(self):
        super(VisualisationNode, self).start(spin=True)

    ############################################################################

    def on_right_image_received(self, img):
        squares_img = self._draw_squares(self._squares, self.right_img)
        self.display_image(squares_img)

    def on_squaremsg_received(self, msg):
        squares = []
        for square_msg in msg.squares:
            squares.append(TrackedSquare.from_msg(square_msg))     

        self._squares = squares   
            
    ############################################################################

    def _draw_squares(self, squares, img):
        for s in self._squares:
            colour_hls = hls_to_rgb(s.hue / 180.0, 0.5, 1.0)
            colour_rgb = [x * 255 for x in colour_hls]
            colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])

            cv2.drawContours(img, np.int0([s.box]), -1, colour_bgr, 3)
            cv2.circle(img, tuple(np.int0(s.center)), 4, colour_bgr, -1)

        return img

####################################################################################################

def main():
    rospy.init_node('visualisation', anonymous=True)

    node = VisualisationNode()
    node.start()

if __name__ == '__main__':
    main()