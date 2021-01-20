import os
os.environ['DISPLAY'] = ':0'
from mtcnn import MTCNN
import cv2
from distance import FacePose


class Detection():
    def __init__(self, img_path):
        self.img_path = img_path
        self.detector = MTCNN()
        

    def detect(self):
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', img)
        return self.detector.detect_faces(img)

if __name__ == "__main__":
    mtcnn_detector = Detection(img_path='./image/one_side.jpg')
    result = mtcnn_detector.detect()[0]
    print(result)
    print(result['keypoints'])
    image = cv2.imread('./image/idol.jpeg')
    image = cv2.rectangle(image, (result['box'][0], result['box'][1]), (result['box'][2], result['box'][3]), (255,0,0), 2)
    
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    distance = FacePose(landmark=result, img_path='./image/one_side.jpg')
    # print('\t', alignment.check_face_pose())
    distance.calc_face_tilt()
    distance.calc_nose_deviation()
