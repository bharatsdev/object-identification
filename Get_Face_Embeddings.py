from imutils.video import VideoStream
from glob import glob
import numpy as np
import imutils
import pickle
import cv2
from fiutils.Utils import Utils

print("[INFO : Load utils.....")
utils = Utils()

face_embeder, label_encoder, face_recognizer, net = utils.load_data()

print("[INFO] : Loading Torch model, It will return the 128-D facial embeddings..... ")
embedder = cv2.dnn.readNetFromTorch(face_embeder)

print("[INFO] : Loading all the images path...")
all_images = glob("data/images/*/*.png")
print(len(all_images))

print("[INFO] : Face Embedding and Labels list")
faceEmbeddings = []
faceLabels = []
total = 0

for idx, imgPath in enumerate(all_images):
    faceName = imgPath.split("\\")[-2]
    image = cv2.imread(imgPath)
    image = imutils.resize(image, width=600)
    h, w = image.shape[:2]

    imgblob = cv2.dnn.blobFromImage(image, 1., (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    net.setInput(imgblob)
    detections = net.forward()
    # ensure at least one face was found
    if len(detections) > 0:
        # Get face with max probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.9:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")
            # Extract Face ROI (Region Of Interest) and get the co-ordinates of it
            face_roi = image[start_y:end_y, start_x:end_x]
            roi_h, roi_w = face_roi[:2]
            # Face Region Should be sufficient 1
            # if roi_h < 20 or roi_w < 20:
            #     continue
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(face_blob)
            embedding_vec = embedder.forward()

            faceLabels.append(faceName)
            faceEmbeddings.append(embedding_vec.flatten())
            total += 1
        print("[INFO]: Embedding completed {}/{} with confidence : {:.5f}% and Name :{}".format(idx, len(all_images),
                                                                                                confidence, faceName))

print("[INFO]: Serialize {} encodings of faces...".format(total))
dataset = {'embeddings': faceEmbeddings, 'names': faceLabels}
f = open("data/output/embeddings.pickle", "wb")
f.write(pickle.dumps(dataset))
f.close()
