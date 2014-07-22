import sys

import cv2

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree

####################################################################################################

class Block(object):
    def __init__(self, abs_pos, rel_pos):
        self.abs_pos = abs_pos
        self.rel_pos = rel_pos

####################################################################################################

class BlockTracker(object):
    _ROI_BORDER = 75

    def __init__(self):
        self._input_size = (400, 640)
        self._roi_rect = (
            BlockTracker._ROI_BORDER, 
            BlockTracker._ROI_BORDER, 
            self._input_size[0] - (BlockTracker._ROI_BORDER * 2), 
            self._input_size[1] - (BlockTracker._ROI_BORDER * 2))

        self._capture = False
        self._current_img = None
        self._last_img = None
        self.display_img = None 

        self.blocks = []

    ############################################################################

    def on_image_received(self, in_img):
        self.detect(in_img)

    ############################################################################

    def detect(self, img):
        roi_img = self._extract_roi(img)
        roi_img_out = self._display_resize(roi_img)

        blocks_mask = self._extract_blocks(roi_img)
        blocks_mask_out = cv2.cvtColor(blocks_mask, cv2.COLOR_GRAY2BGR)
        blocks_mask_out = self._display_resize(blocks_mask_out)

        self._detect_blocks(blocks_mask)
        blocks_img = self._draw_blocks()
        blocks_img_out = self._display_resize(blocks_img)

        display_img_out = cv2.addWeighted(img, 0.25, roi_img_out, 0.75, 0)
        display_img_out = cv2.addWeighted(display_img_out, 1.0, blocks_mask_out, 0.25, 0)
        display_img_out = cv2.addWeighted(display_img_out, 1.0, blocks_img_out, 1.0, 0)

        self.display_img = display_img_out

    def _detect_blocks(self, blocks_img):
        lap_img = cv2.Laplacian(blocks_img, cv2.CV_8U)
        contours, _ = cv2.findContours(lap_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_blocks = []
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            box = np.int0(cv2.cv.BoxPoints(rect))

            try:
                M = cv2.moments(hull)
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']

                abs_pos = (cx, cy)
                
                rel_pos = (
                    ((cx / blocks_img.shape[1]) - 0.5) * 2,
                    ((cy / blocks_img.shape[0]) - 0.5) * 2
                    )

                block = Block(abs_pos, rel_pos)
                detected_blocks.append(block)
            except:
                continue

        self.blocks = detected_blocks

    def _extract_blocks(self, in_img):
        yellow_img, yellow_mask = self._select_hue(in_img, 25)        
        green_img, green_mask = self._select_hue(in_img, 45)
        red_img, red_mask = self._select_hue(in_img, 5)
        blocks_img = yellow_img + green_img + red_img
        blocks_mask = yellow_mask + green_mask + red_mask

        blocks_mask = cv2.medianBlur(blocks_mask, 17)
        return blocks_mask

    def _draw_blocks(self):
        img = self._new_roi_array()

        for block in self.blocks:
            x = int(((block.rel_pos[0] / 2.0) + 0.5) * img.shape[1])
            y = int(((block.rel_pos[1] / 2.0) + 0.5) * img.shape[0])
            cv2.rectangle(img, (x - 5, y - 5), (x  + 5, y + 5), (255, 0, 255))

        return img

    ############################################################################

    def _select_hue(self, in_img, hue, 
            hue_range=10, 
            lower_sv_range=(70, 30), 
            upper_sv_range=(255, 255)):
        hsv_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)

        lower_range = np.array([hue - hue_range, lower_sv_range[0], lower_sv_range[1]])
        upper_range = np.array([hue + hue_range, upper_sv_range[0], upper_sv_range[1]])
        mask = cv2.inRange(hsv_img, lower_range, upper_range)
        
        return (cv2.bitwise_and(in_img, in_img, mask=mask), mask)

    def _extract_roi(self, img, rect=None):
        if rect is None:
            rect = self._roi_rect

        roi_img = img[
            rect[0]:rect[0] + rect[2],
            rect[1]:rect[1] + rect[3],
            :]

        return roi_img

    def _display_resize(self, img, to_size=None, to_rect=None):
        if to_size is None:
            to_size = self._input_size
        if to_rect is None:
            to_rect = self._roi_rect

        img_out = np.zeros((to_size[0], to_size[1], 3), dtype='uint8')
        img_out[
            to_rect[0]:to_rect[0] + img.shape[0],
            to_rect[1]:to_rect[1] + img.shape[1],
            :] = img

        return img_out

    ############################################################################

    def _new_roi_array(self):
        return np.zeros((self._roi_rect[2], self._roi_rect[3], 3), dtype='uint8')