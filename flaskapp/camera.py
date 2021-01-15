import cv2
import numpy as np
from flaskapp.model import FaceExpressionModel
from flaskapp.model import FaceDetectionModel
from flaskapp.model import FaceMaskDetectionModel
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

face_detect = FaceDetectionModel('flaskapp/static/face_detector/deploy.prototxt', 'flaskapp/static/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
expression_detect = FaceExpressionModel('flaskapp/static/emotion_detector/model.json','flaskapp/static/emotion_detector/model_weights.h5')
mask_detect = FaceMaskDetectionModel('flaskapp/static/mask_detector/mask_detector.model')
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes 
    def get_frame(self):
        _, frame = self.video.read()
        locs = face_detect.predict_facial_regions(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = facec.detectMultiScale(gray_frame, 1.3, 5)
        
        faces = []
        mask_preds = []
        emotion_preds = []
        gray_frames = []

        if len(locs) > 0:
            for (startX, startY, endX, endY) in locs:
                frame_crop = frame[startX:endX, startY:endY]
                gray_frame = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
                try:
		    #faces = facec.detectMultiScale(gray_frame, 1.3, 5)
                    face = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    gray_frame = cv2.resize(gray_frame, (48, 48))
                    emotion_pred = expression_detect.predict_emotion(gray_frame[np.newaxis, :, :, np.newaxis])
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    emotion_preds.append(emotion_pred)
                    faces.append(face)
                except Exception as e:
                    print(str(e))
		        #print(frame_crop.shape, "\t", face.shape)
            
            
            #cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            mask_preds = mask_detect.predict_masked_faces(faces)

        for(box, mask_pred, emotion_Pred) in zip(locs, mask_preds, emotion_preds):
            # unpack the bounding box and predictions
            #print(box.shape, "\t", box)
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = mask_pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
            cv2.putText(frame, label, (startX, startY - 10),
              	font, 0.45, color, 2)
            cv2.putText(frame, emotion_Pred, (startX, startY + 20),
              	font, 0.65, (255, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
