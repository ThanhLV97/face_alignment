import matplotlib.pyplot as plt
import math
import numpy as np


class Alignment():
    def __init__(self, landmark=None):
        """This module used to align face image and evaluate face pose of face   image

        Args:
            landmark ([type], optional): Five point of face landmark. Defaults to None.
        """
        self.landmark = landmark


    def evaluate(self):
        d_left_eye = self._calc_distance(self.landmark['nose'], self.landmark['mouth_left'], self.landmark['left_eye'])
        d_right_eye = self._calc_distance(self.landmark['nose'], self.landmark['mouth_left'], self.landmark['right_eye'])
        if d_left_eye > d_right_eye:
            ratio = d_right_eye / d_left_eye
        elif d_left_eye < d_right_eye:
            ratio = d_left_eye / d_right_eye
        else:
            ratio = 1
        return ratio


    def _calc_distance(self, point_a, point_b, point_c):
        pa = np.asarray(point_a)
        pb = np.asarray(point_b)
        pc = np.asarray(point_c)
        return np.cross(pb-pa,pc-pa)/np.linalg.norm(pb-pa)
        