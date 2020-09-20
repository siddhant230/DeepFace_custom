import os
import cv2
import math
import base64
import falcon
import logging
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)


class FaceExtract:

    def __init__(self):
        try:
            face_detector_path = "./lib/deepface_custom/haar_xmls/haarcascade_frontalface_default.xml"
            eye_detector_path = "./lib/deepface_custom/haar_xmls/haarcascade_eye.xml"
            self.face_detector = cv2.CascadeClassifier(face_detector_path)
            self.eye_detector = cv2.CascadeClassifier(eye_detector_path)
            logger.info("face and eye detector models loaded")
        except:
            logger.error("Unable to load face or eye detector models")
            raise falcon.HTTPBadRequest(title="Unable to load face or eye detector models")

        try:
            prototxt_path = './lib/deepface_custom/face_detector/deploy.prototxt'
            weights_path = './lib/deepface_custom/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
            self.face_ext_model = cv2.dnn.readNet(prototxt_path, weights_path)
            logger.info('face Extractor model loaded')
        except:
            logger.error("Unable to load face extractor model")
            raise falcon.HTTPBadRequest(title="Unable to load face extractor model")

    def detect_face_mod(self, frame: np.array, confidence_thresh=0.5):
        """
        :params img: (np array) selfie image
        : return extracted face
        """

        frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.face_ext_model.setInput(blob)
        detections = self.face_ext_model.forward()

        faces = []
        locs = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_thresh:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]
                try:
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)
                except:
                    logger.info("selfie face not detected")
                    raise falcon.HTTPBadRequest(title="Selfie face not detected!")

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            sx, sy, ex, ey = locs[-1]

            return frame[sy:ey, sx:ex]
        else:
            return []

    @staticmethod
    def distance_point(a: np.array, b: np.array):
        """
        Receives the images, extracts faces and compares the similarity

        :param a: (np array) point set 1
        :param b: (np array) point set 2
        :return:
        """
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]

        return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

    @staticmethod
    def get_blurred(img: np.array):
        """
        Checks if the image is too blurred to be processed

        :param img: (np array) face image
        :return: result: (boolean) return True if blurriness is high
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_kernel_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_kernel_value

    def detect_face(self, img: np.array, grayscale=False):
        """
        :params img: (np array) doc image
        : return extracted face
        """
        img_raw = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        buffer = 40
        if len(faces) > 0:
            # getting sharpest face in doc
            sharpest = -1
            sharpness_val = 0
            for index in range(len(faces)):
                x, y, w, h = faces[index]
                current_face = img[int(y):int(y + h), int(x):int(x + w)]
                current_face = cv2.resize(current_face, (160, 160), interpolation=cv2.INTER_AREA)
                current_value = self.get_blurred(current_face)
                if current_value > sharpness_val:
                    sharpest = index
                    sharpness_val = current_value
            x, y, w, h = faces[sharpest]

            points = [x, y, w, h]
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]
            a, b, c, d = max(0, y - buffer), y + h + 2 * buffer, max(0, x - buffer), x + w + 2 * buffer
            buffer_face = img[max(0, y - buffer):y + h + 2 * buffer, max(0, x - buffer):x + w + 2 * buffer]
            detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_detector.detectMultiScale(detected_face_gray)

            if len(eyes) >= 2:
                # find the largest 2 eye
                base_eyes = eyes[:, 2]

                items = []
                for i in range(0, len(base_eyes)):
                    item = (base_eyes[i], i)
                    items.append(item)

                df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(by=['length'], ascending=False)

                eyes = eyes[df.idx.values[0:2]]

                # -----------------------
                # decide left and right eye

                eye_1 = eyes[0]
                eye_2 = eyes[1]

                if eye_1[0] < eye_2[0]:
                    left_eye = eye_1
                    right_eye = eye_2
                else:
                    left_eye = eye_2
                    right_eye = eye_1

                # -----------------------
                # find center of eyes

                left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
                left_eye_x = left_eye_center[0]
                left_eye_y = left_eye_center[1]

                right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
                right_eye_x = right_eye_center[0];
                right_eye_y = right_eye_center[1]

                # -----------------------
                # find rotation direction

                if left_eye_y > right_eye_y:
                    point_3rd = (right_eye_x, left_eye_y)
                    direction = -1  # rotate same direction to clock
                else:
                    point_3rd = (left_eye_x, right_eye_y)
                    direction = 1  # rotate inverse direction of clock

                # -----------------------
                # find length of triangle edges

                a = self.distance_point(left_eye_center, point_3rd)
                b = self.distance_point(right_eye_center, point_3rd)
                c = self.distance_point(right_eye_center, left_eye_center)

                # -----------------------
                # apply cosine rule

                if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

                    cos_a = (b * b + c * c - a * a) / (2 * b * c)
                    angle = np.arccos(cos_a)  # angle in radian
                    angle = (angle * 180) / math.pi  # radian to degree
                    if direction == -1:
                        angle = 90 - angle

                    img = Image.fromarray(img_raw)
                    img = np.array(img.rotate(direction * angle))

                    faces = self.face_detector.detectMultiScale(img, 1.3, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        points = [x, y, w, h]
                        a, b, c, d = max(0, y - buffer), y + h + 2 * buffer, max(0, x - buffer), x + w + 2 * buffer
                        buffer_face = img[max(0, y - buffer):y + h + 2 * buffer, max(0, x - buffer):x + w + 2 * buffer]

                        detected_face = img[int(y):int(y + h), int(x):int(x + w)]

            if grayscale:
                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

            return detected_face, points, buffer_face, [a, b, c, d]

    def get_face(self, img: np.array, status=1):
        """
        :params img: (np array) doc image or selfie image
        : return extracted face
        """

        if status == 0:  # selfie
            try:
                face = self.detect_face_mod(img)
                logger.info("Selfie face detected")
                return np.array(face * 255, dtype=np.uint8)
            except:
                logger.error('Selfie face not detected!')
                return []
        else:  # docs
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (600, 400))
            img = cv2.filter2D(img, -1, kernel)
            try:
                face, _, _, _ = self.detect_face(img)
                logger.info('Doc face detected!')
                return np.array(face * 255, dtype=np.uint8)
            except:
                logger.error('Doc face not detected!')
                return []
