#!/usr/bin/env python

import rospy
from glasgow_baxter_helpers import BaxterNode
from glasgow_baxter.msg import DetectedSquares, TrackedSquares
from square import Square, TrackedSquare

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import MeanShift, AffinityPropagation, DBSCAN, estimate_bandwidth

import random
import itertools
import collections

####################################################################################################

class UnderstandingNode(BaxterNode):
    def __init__(self):
        super(UnderstandingNode, self).__init__()

        self._squares_sub = rospy.Subscriber(
            '/squares', 
            DetectedSquares,
            self.on_squaremsg_received)

        self._squares_pub = rospy.Publisher(
            '/tracked_squares', 
            TrackedSquares,
            tcp_nodelay=True)

        self._prev_squares = collections.deque(maxlen=20)

    ############################################################################

    def start(self):
        super(UnderstandingNode, self).start(spin=True)

    ############################################################################

    def on_squaremsg_received(self, msg):
        detected_squares = []
        for square_msg in msg.squares:
            detected_squares.append(TrackedSquare.from_msg(square_msg))

        self._prev_squares.append(detected_squares)
        
        all_squares = list(itertools.chain.from_iterable(self._prev_squares))
        square_centers = [list(s.center) + [s.hue] for s in all_squares]
        data = np.array(square_centers)

        ms = DBSCAN(eps=64, min_samples=3)
        ms.fit(data)
        labels = ms.labels_

        ts_msg = TrackedSquares()
        for i, s in enumerate(all_squares):
            label = np.int0(labels[i])
            if label < 0: 
                continue

            s.tracking_colour = TrackedSquare.TRACKING_COLOURS[label % len(TrackedSquare.TRACKING_COLOURS)]
            s.tracking_detected = True

            ts_msg.squares.append(s.to_msg())

        self._squares_pub.publish(ts_msg)


    ############################################################################

    def _track_squares(self, detected_squares):
        if self._tracked_squares is None:
            self._tracked_squares = dict(map(lambda s: (s.tracking_id, s), detected_squares))
            return

        min_squares = self._match_min_squares(detected_squares, self._tracked_squares.values())

        # Update tracking with the new square.
        for ds, ts in min_squares.items():
            self._tracked_squares[ts.tracking_id] = ds

            ds.tracking_detected = True
            ds.tracking_id = ts.tracking_id
            ds.tracking_colour = ts.tracking_colour

        # Mark any untracked squares this frame as inactive.
        ts_msg = TrackedSquares()
        for ts in self._tracked_squares.values():
            if not ts in min_squares.keys():
                ts.tracking_detected = False

        self._publish_tracked_squares()

    ############################################################################

    def _build_distance_matrix(self, detected_squares, tracked_squares):
        # Calculate distances between tracked squares and new squares.
        distance_matrix = {}
        for ds in detected_squares:
            distance_matrix[ds] = {}
            for ts in tracked_squares: 
                distance_matrix[ds][ts] = distance.minkowski(ds.center, ts.center, 128)

        return distance_matrix

    def _sort_squares_by_distance_matrix(self, distance_matrix):
        # Calculate distances between tracked squares and new squares.
        sorted_squares = {}
        for ds in distance_matrix.keys():
            sorted_squares[ds] = collections.deque(
                sorted(distance_matrix[ds].keys(), key=lambda ts: distance_matrix[ds][ts]))

        return sorted_squares

    def _match_min_squares(self, detected_squares, tracked_squares):
        distance_matrix = self._build_distance_matrix(detected_squares, 
            self._tracked_squares.values())
        sorted_squares = self._sort_squares_by_distance_matrix(distance_matrix)

        min_squares_ts_to_ds = {}
        min_squares_ds_to_ts = {}
        need_matched = collections.deque(distance_matrix.keys())
        while len(need_matched) > 0:
            ds = need_matched.popleft()

            # Leave it unmatched if there's nothing else.
            if not len(sorted_squares[ds]) > 0:
                continue

            min_ts = sorted_squares[ds].popleft()

            if not min_ts in min_squares_ts_to_ds:
                # Tracked square is unmatched, so match it right away.
                min_squares_ds_to_ts[ds] = min_ts
                min_squares_ts_to_ds[min_ts] = ds
            else:
                # Closest tracked square has already been tracked.
                ds2 = min_squares_ts_to_ds[min_ts]

                # Check which one is closest.
                if distance_matrix[ds][min_ts] < distance_matrix[ds2][min_ts]:
                    # This one is closest, so remap.
                    min_squares_ds_to_ts[ds] = min_ts
                    min_squares_ts_to_ds[min_ts] = ds

                    del min_squares_ds_to_ts[ds2]
                    need_matched.append(ds2)
                else:
                    # Otherwise, try again later.
                    need_matched.append(ds)

        return min_squares_ds_to_ts

    def _publish_tracked_squares(self):
        # Output all detected squares.
        ts_msg = TrackedSquares()
        for s in self._tracked_squares.values():
            ts_msg.squares.append(s.to_msg())
        self._squares_pub.publish(ts_msg)

####################################################################################################

def main():
    rospy.init_node('understanding', anonymous=True)

    node = UnderstandingNode()
    node.start()

if __name__ == '__main__':
    main()