import cv2
import json
import falcon
import logging
import Facenet
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)


class DeepFaceCustom:
    def __init__(self):
        try:
            self.face_net_model = Facenet.loadModel()
            logger.info('FaceNet model loaded')
        except:
            logger.info('FaceNet model not loaded')
            
    @staticmethod
    def find_threshold(distance_metric: str):
        """
        :params distance_metric: (str) specifies the type of distance metric
        : return threshold value for specific metric
        """
        if distance_metric == 'cosine':
            threshold = 0.40
        elif distance_metric == 'euclidean':
            threshold = 10
        elif distance_metric == 'euclidean_l2':
            threshold = 0.80

        return threshold

    @staticmethod
    def find_cosine_distance(source_representation: np.array, test_representation: np.array):
        """
        :params source_representation: (np array) vector representation of source image
        :params test_representation: (np.array) vector representation of test image
        : return cosine distance
        """
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def find_euclidean_distance(source_representation: np.array, test_representation: np.array):
        """
        :params source_representation: (np array) vector representation of source image
        :params test_representation: (np.array) vector representation of test image
        : return euclidean distance
        """
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    @staticmethod
    def l2_normalize(x: np.array):
        """
        :params x: un-normalized state vector x
        : return normalized x
        """
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def verify_faces(self, image_1, image_2, shape=(160, 160), distance_metric='cosine'):
        """
        Receives the images, extracts faces and compares the similarity

        :param image_1: (np array) raw RGB image from document
        :param image_2: (np array) raw RGB image from selfie
        :param shape: (tuple) expected shape of images
        :param distance_metric: (np array) type of distance to be used for comparison
        :return: json results of predictions
        """

        image_1 = cv2.resize(image_1, shape, interpolation=cv2.INTER_AREA)
        image_2 = cv2.resize(image_2, shape, interpolation=cv2.INTER_AREA)

        img1 = image_1.reshape(1, *image_1.shape)
        img2 = image_2.reshape(1, *image_2.shape)

        threshold = self.find_threshold(distance_metric)
        img1 = img1.copy()
        img2 = img2.copy()
        img1_representation = self.face_net_model.predict(img1)[0, :]
        img2_representation = self.face_net_model.predict(img2)[0, :]
        if distance_metric == 'cosine':
            distance = self.find_cosine_distance(img1_representation, img2_representation)
        elif distance_metric == 'euclidean':
            distance = self.find_euclidean_distance(img1_representation, img2_representation)
        else:
            raise ValueError("Invalid distance_metric passed - ", distance_metric)

        if distance <= threshold:
            identified = "true"
        else:
            identified = "false"

        resp_obj = "{"
        resp_obj += "\"verified\": " + identified
        resp_obj += ", \"distance\": " + str(distance)
        resp_obj += "}"

        resp_obj = json.loads(resp_obj)

        return resp_obj
