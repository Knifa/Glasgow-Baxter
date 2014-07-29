#!/usr/bin/env python

import rospy
from glasgow_baxter_helpers import BaxterNode

import numpy as np
import cv2
import cv2.cv as cv
from scipy.spatial import distance

import os

################################################################################

class FaceTrackerNode(BaxterNode):
    CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'
    CLASSIFIER_DIR = '/usr/share/opencv/haarcascades/'
    SCALE = 0.5

    RATE = 5

    TURN_THRESHOLD = 32
    TURN_SPEED = (1.0 / RATE) * (np.pi / 4.0)

    ############################################################################

    def __init__(self):
        super(FaceTrackerNode, self).__init__()

        self._cc = cv2.CascadeClassifier(
            os.path.join(self.CLASSIFIER_DIR, self.CLASSIFIER_FILE))
        
    ############################################################################

    def start(self):
        super(FaceTrackerNode, self).start()

        while self.head_img is None:
            pass

        self.head.set_pan(0)

        r = rospy.Rate(self.RATE)
        while not rospy.is_shutdown():
            rects = self._detect_faces(self.head_img)
            centers = self._pan_head_to_nearest_face(rects, self.head_img.shape)

            img = self._draw_faces(self.head_img, rects, centers)

            self.display_image(img)
            r.sleep()

    ############################################################################

    def _pan_head_to_nearest_face(self, rects, img_shape):
        centers = []
        screen_centers = []
        for r in rects:
            center = (r[:2] + r[2:]) / 2
            center_origin = center - (np.array(img_shape)[1::-1] / 2)

            centers.append(center_origin)
            screen_centers.append(center)

        if len(centers) > 0:
            min_center = min(centers, key=lambda x: distance.euclidean(x, [0,0]))
            move_scale = np.clip(distance.euclidean(min_center, [0,0]) / (self.TURN_THRESHOLD*4), 0.02, 1.0)

            new_angle = self.head.pan()
            if min_center[0] <= -(self.TURN_THRESHOLD / 2):
                new_angle -= self.TURN_SPEED * move_scale
            elif min_center[0] >= (self.TURN_THRESHOLD / 2):
                new_angle += self.TURN_SPEED * move_scale

            self.head.set_pan(new_angle, timeout=0.0)

        return screen_centers

    def _detect_faces(self, img):
        img = self._filter_image(img)
        rects = self._cc.detectMultiScale(img, scaleFactor=1.25)

        if len(rects) == 0:          
            return []      
        else:
            rects[:,2:] += rects[:,:2]
            rects[:,:] = np.int0(rects[:,:] / self.SCALE)

        return rects

    def _draw_faces(self, img, rects, centers):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for x, y in centers:
            cv2.rectangle(
                img, 
                (x - 5, y - 5), 
                (x + 5, y + 5), 
                (255, 255, 255), 
                2)

        return img

    ############################################################################

    def _filter_image(self, img):
        img = self._sharpen(img)
        img = cv2.resize(
            img, 
            (int(img.shape[1] * self.SCALE), int(img.shape[0] * self.SCALE)),
            interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)

        return img

    def _sharpen(self, img):
        blur_img = cv2.GaussianBlur(img, (5, 5), 5)
        sharp_img = cv2.addWeighted(img, 2.0, blur_img, -1.0, 0)

        return sharp_img

################################################################################

if __name__ == '__main__':
    rospy.init_node('face_tracker')

    n = FaceTrackerNode()
    n.start()