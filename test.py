from mtcnn import MTCNN
import cv2
from distance import Alignment


class Detection():
    def __init__(self, img_path):
        self.img_path = img_path
        self.detector = MTCNN()
        self.alignment = Alignment()

    def detect(self):
        img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', img)
        return self.detector.detect_faces(img)

if __name__ == "__main__":
    mtcnn_detector = Detection(img_path='idol.jpeg')
    result = mtcnn_detector.detect()[0]
    print(result['keypoints'])
    alignment = Alignment(landmark=result['keypoints'])
    print(alignment.evaluate())


