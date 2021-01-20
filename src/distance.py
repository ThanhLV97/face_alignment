import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


class FacePose():
    def __init__(self, landmark, img_path=None):
        """This module used to align face image and evaluate face pose of face   image

        Args:
            landmark ([type], optional): Five point of face landmark. Defaults to None.
        """
        self.landmark = landmark['keypoints']
        self.box = landmark['box']
        self.img = cv2.imread(img_path)
        self.head_tilt = None
        self.nose_deviation = None




    def _calc_distance(self, point_a, point_b, point_c):
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


    def calc_face_tilt(self):
        """ Measure angle according to eyes cordinate"""

        # Draw line between left eye and right eye
        # TODO: Remove draw
        img = cv2.line(self.img, self.landmark['left_eye'],\
                    self.landmark['right_eye'], color=(255, 0, 0), thickness=2)

        img = cv2.circle(img, self.landmark['left_eye'],\
                            radius=0, color=(0, 0, 255), thickness=2)
        img = cv2.circle(img, self.landmark['nose'], \
                            radius=0, color=(0, 0, 255), thickness=10)
        img = cv2.circle(img, self.landmark['mouth_left'], \
                            radius=0, color=(0, 0, 255), thickness=10)
        img = cv2.circle(img, self.landmark['mouth_right'], \
                            radius=0, color=(0, 0, 255), thickness=10)
        img = cv2.rectangle(img, (self.box[0], self.box[1]), (self.box[0] + self.box[2], self.box[1] + self.box[3]), color=(255, 0, 0), thickness=4)


        cv2.imshow('image', img)
        cv2.waitKey(0)

        if self.landmark['left_eye'][1] < self.landmark['right_eye'][1]:
            y = self.landmark['right_eye'][1] - self.landmark['left_eye'][1]
            x = self.landmark['right_eye'][0] - self.landmark['left_eye'][0]
            self.tilt_angle = np.arctan2(y, x)
            print(self.tilt_angle)

        else:
            y = self.landmark['left_eye'][1] - self.landmark['right_eye'][1]
            x = self.landmark['right_eye'][0] - self.landmark['left_eye'][0]
            self.tilt_angle = np.arctan2(y, x)
            print(self.tilt_angle)

        return self.tilt_angle


    def calc_nose_deviation(self):
        """ Calculate nose deviation according to distance ratio

        Returns:
            [float]: Ratio
        """
        # calc x box center
        x_center = self.box[0] + self.box[3]/2
        nose_center_dis = abs(self.landmark['nose'][0] - x_center)
        self.nose_deviation = nose_center_dis / (self.box[2]/2) * 0.8
        print('nose_distance', nose_center_dis)
        print('center', self.box[2]/2)
        print('nose distance ', self.nose_deviation)
        return self.nose_deviation
