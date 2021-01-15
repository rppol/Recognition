from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.15
# session = tf.compat.v1.Session(config=config)


class FaceDetectionModel(object):
    def __init__(self, prototxt_file, caffemodel_file):
        print("[INFO] loading face detector model...")
        self.face_detect_model = cv2.dnn.readNet(prototxt_file, caffemodel_file)

    def predict_facial_regions(self, frame):
        #print(frame)
        (self.h, self.w) = frame.shape[:2]
        self.blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.face_detect_model.setInput(self.blob)
        self.detections = self.face_detect_model.forward()

        faces = []
        locs = []

        # loop over the detections
        for i in range(0, self.detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            self.confidence = self.detections[0, 0, i, 2]


            if self.confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
	        # the object
                self.box = self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (self.startX, self.startY, self.endX, self.endY) = self.box.astype("int")

                # ensure the bounding boxes fall within the dimensions of the frame
                (self.startX, self.startY) = (max(0, self.startX), max(0, self.startY))
                (self.endX, self.endY) = (min(self.w - 1, self.endX), min(self.h - 1, self.endY))

                locs.append((self.startX, self.startY, self.endX, self.endY))

            '''if len(locs) > 0:
                for (self.startX, self.startY, self.endX, self.endY) in locs:
                    frame_crop = frame[self.startX:self.endX, self.startY:self.endY]
                    gray_frame = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
		    #faces = facec.detectMultiScale(gray_frame, 1.3, 5)
                    face = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    faces.append(face)'''
        return locs



class FaceMaskDetectionModel(object):
    def __init__(self, model_file):
        print("[INFO] loading face mask detector model...")
        self.mask_detect_model = load_model(model_file)

    def predict_masked_faces(self, faces):
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            self.preds_mask = self.mask_detect_model.predict(faces, batch_size=32)
        return self.preds_mask

class FaceExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FaceExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
