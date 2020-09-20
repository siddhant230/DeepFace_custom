import io
import cv2
import falcon
import base64
import logging
import numpy as np
from PIL import Image
from DeepFace_custom import DeepFaceCustom as dfc
from Face_extract import FaceExtract as fe

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)


class FaceSimilarity(dfc, fe):
    def __init__(self):
        """
        Loading model for document localization

        :param model: (binary h5) The saved model for document localization
        """
        fe.__init__(self)
        dfc.__init__(self)
        

    def verify(self, face_1: np.array, face_2: np.array):
        """
        Verify the similarity between faces by using DeepFace algo

        :param face_1: (np array) face image from document
        :param face_2: (np array) face image from selfie
        :return: result: (float) distance value between faces
        """
        result = self.verify_faces(face_1, face_2, distance_metric='cosine')
        return result

    @staticmethod
    def is_blurred(img: np.array):
        """
        Checks if the image is too blurred to be processed

        :param img: (np array) face image
        :return: result: (boolean) return True if blurriness is high
        """
        threshold = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_kernel_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        result = laplacian_kernel_value < threshold

        return result

    def find_similarity(self, face_1: np.array, face_2: np.array):
        """
        Receives the images, extracts faces and compares the similarity

        :param face_1: (np array) raw RGB image from document
        :param face_2: (np array) raw RGB image from selfie
        :return:
        """
        try:
            if not self.is_blurred(face_1) and not self.is_blurred(face_2):
                result = self.verify(face_1, face_2)
                return result
            else:
                return None
        except:
            logger.info("Cannot find face")

    def match_faces(self, image_1: np.array, image_2: np.array):
        """
        Processes image urls, corrects orientation and finds similarity between images

        :param image_doc_url_1: (str) face image url from document
        :param image_doc_url_2: (str) face image url from document
        :param image_selfie_url: (str) face image url from selfie
        :return: result: (float) distance value between faces
        """

        face_1 = []
        if len(image_1) > 0:
            corrected_image1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            face_1 = self.get_face(img=corrected_image1, status=0)  # doc : 1 and selfie : 0

        face_2 = []
        if len(image_2) > 0:
            corrected_image2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
            face_2 = self.get_face(img=corrected_image2, status=0)  # doc : 1 and selfie : 0

        if len(face_1) == 0 and len(face_2) == 0:
            logger.error('Face not detected in any document!')

        print(face_1.shape, face_2.shape)
        result = None
        if len(face_1) != 0:
            result = self.find_similarity(face_1, face_2)
            return result
        return None


face_match_obj = FaceSimilarity()
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()

    face1 = img.copy()
    face2 = img.copy()

    result = face_match_obj.match_faces(face1, face2)
    
    print(result)
    cv2.imshow("im1", face1)
    cv2.imshow('im2', face2)
    if cv2.waitKey(1) == ord('q'):
        break
