import cv2


class Utils:
    def __init__(self):
        print("[INFO] : Utils object created")

    def load_data(self):
        print("[INFO] : Loading SSD Face Detector from desk.... ")
        prototxt = "data/recognition_model/deploy.prototxt"
        caffemodel = "data/recognition_model/res10_300x300_ssd_iter_140000.caffemodel"
        face_embeder = "data/recognition_model/openface.nn4.small2.v1.t7"
        label_encoder = "data/output/labelencoder.pickle"
        face_recognizer = "data/output/facerecognizer.pickle"
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        return face_embeder, label_encoder, face_recognizer, net
