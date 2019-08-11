import numpy as np
import argparse
import imutils
import pickle
import cv2
from glob import glob
from fiutils.Utils import Utils

print("[INFO : Load utils.....")
utils = Utils()

face_embeder, label_encoder, face_recognizer, net = utils.load_data()

print("[INFO] : Loading Torch model, It will give the 128-D facial embeddings..... ")
embedder = cv2.dnn.readNetFromTorch(face_embeder)

print("[INFO] : Load Serialized Face Label Encoder....")
faceRecognizer = pickle.loads(open(face_recognizer, "rb").read())
faceLabelEncoder = pickle.loads(open(label_encoder, "rb").read())

print("[INFO] : Identifing the faces from img...")
print("[INFO] : Loading all the images path...")
all_images = glob("data/testimg/*.png")
print("[INFO] : Number of Test Sample ", len(all_images))

for idx, imgPath in enumerate(all_images):
    image = cv2.imread(imgPath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    img_blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    net.setInput(img_blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > 0.9:
            print(confidence)
            fac_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = fac_box.astype("int")
            # Extract Face ROI (Region Of Interest) and get the co-ordinates of it
            face_roi = image[start_y:end_y, start_x:end_x]
            roi_h, roi_w = face_roi[:2]
            # Face Region Should be sufficient 1
            # if roi_h < 20 or roi_w < 20:
            #     continue
            print("[INFO] : Get the face embedding from ..... ")
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(face_blob)
            embedding_vec = embedder.forward()

            print("[INFO] : Perform the predictios on the faces...")
            preds = faceRecognizer.predict_proba(embedding_vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            faceName = faceLabelEncoder.classes_[j]

            print("[INFO] : Drawing bounding box of face with probabilities...")
            text = "{} : {:.3f}%".format(faceName, proba * 100)
            labelCoordinates = start_y
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(image, text, (abs(start_x), abs(start_y)), cv2.FONT_ITALIC, 0.45, (100, 100, 100), 2)
    cv2.imshow("Face-Identi", image)
    cv2.waitKey(0)
