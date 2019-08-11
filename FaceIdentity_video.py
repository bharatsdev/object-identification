import numpy as np
import imutils
import time
import pickle
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
from fiutils.Utils import Utils

print("[INFO : Load utils.....")
utils = Utils()

face_embeder, label_encoder, face_recognizer, net = utils.load_data()

print("[INFO] : Loading Torch model, It will give the 128-D facial embeddings..... ")
embedder = cv2.dnn.readNetFromTorch(face_embeder)

print("[INFO] : Load Serialized Face Label Encoder....")
faceRecognizer = pickle.loads(open(face_recognizer, "rb").read())
faceLabelEncoder = pickle.loads(open(label_encoder, "rb").read())

print("[INFO] : Start video Stream......")
vcap = VideoStream(src=0).start()
time.sleep(0.2)

print("[INFO] : FPS through rate....")
fps = FPS().start()
print(fps)

while True:
    frame = vcap.read()
    resizeImg = imutils.resize(frame, width=600)
    h, w = resizeImg.shape[:2]
    frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                       (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(frame_blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > 0.9:
            print(confidence)
            fac_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = fac_box.astype("int")
            # Extract Face ROI (Region Of Interest) and get the co-ordinates of it
            face_roi = frame[start_y:end_y, start_x:end_x]
            roi_h, roi_w = face_roi[:2]
            print("[INFO] : Get the face embedding from ..... ")
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(face_blob)
            embedding_vec = embedder.forward()

            print("[INFO] : Perform the predictios on the faces...")
            preds = faceRecognizer.predict_proba(embedding_vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            faceName = faceLabelEncoder.classes_[j]
            if proba > 0.95:
                print("[INFO] : Drawing bounding box of face with probabilities...")
                text = "{} : {:.3f}%".format(faceName, proba * 100)
                labelCoordinates = start_y
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                cv2.putText(frame, text, (abs(start_x), abs(start_y)), cv2.FONT_ITALIC, 0.45, (100, 100, 100), 2)
                # update the FPS counter
    fps.update()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vcap.stop()
