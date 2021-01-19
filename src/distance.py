import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


class Alignment():
    def __init__(self, landmark=None, img_path=None):
        """This module used to align face image and evaluate face pose of face   image

        Args:
            landmark ([type], optional): Five point of face landmark. Defaults to None.
        """
        self.landmark = landmark
        self.img = cv2.imread(img_path)
        self.pose_angle = None



    def check_face_pose(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        mid_mouth = (round((self.landmark['mouth_left'][0] + self.landmark['mouth_right'][0])/2), \
                        round((self.landmark['mouth_right'][1] + self.landmark['mouth_left'][1])/2))
        
        d_left_eye = abs(self._calc_distance(self.landmark['nose'], mid_mouth, self.landmark['left_eye']))
        d_right_eye = abs(self._calc_distance(self.landmark['nose'], mid_mouth, self.landmark['right_eye']))
        d_test = self._calc_distance((1, 2), (1,4), (3, 3))
        
        if d_left_eye > d_right_eye:
            ratio = d_right_eye / d_left_eye
        elif d_left_eye < d_right_eye:
            ratio = d_left_eye / d_right_eye
        else:
            ratio = 1
        return ratio


    def _measure_distance(self, point_a, point_b, point_c):
        """ This function to calculate distance from a point to the line which created by two points

        Args:
            point_a ([tuple]): a point of the line
            point_b ([type]): other point of the line
            point_c ([type]): point to calculate distance

        Returns:
            float : Distance from point_c to the line.
        """
        pa = np.asarray(point_a)
        pb = np.asarray(point_b)
        pc = np.asarray(point_c)
        return np.cross(pb-pa, pc-pa) / np.linalg.norm(pb-pa)
    
    def measure_angle_eyes(self):
        """ Measure angle according to eyes position"""

        # Draw line between left eye and right eye
        img = cv2.line(self.img, self.landmark['left_eye'],\
                    self.landmark['right_eye'], color=(255, 0, 0), thickness=2)
        
        img = cv2.circle(img, self.landmark['left_eye'], radius=0, color=(0,0,255), thickness=2)
        img = cv2.circle(img, self.landmark['nose'], radius=0, color=(0, 0, 255), thickness=10)


        if self.landmark['left_eye'][1] < self.landmark['right_eye'][1]:
            y = self.landmark['right_eye'][1] - self.landmark['left_eye'][1]
            x = self.landmark['right_eye'][0] - self.landmark['left_eye'][0]
            self.pose_angle = np.arctan2(y, x)
            print(self.pose_angle)

        else:
            y = self.landmark['left_eye'][1] - self.landmark['right_eye'][1]
            x = self.landmark['right_eye'][0] - self.landmark['left_eye'][0]
            self.pose_angle = np.arctan2(y, x)
            print(self.pose_angle)
