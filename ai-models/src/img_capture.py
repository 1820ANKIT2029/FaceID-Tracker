import os
import cv2
from matplotlib import pyplot as plt
import uuid

from .config import config

def get_video_frame():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        frame = frame[120:120+250, 200:200+250]

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

        cv2.imshow('Image Collection', frame)

    return frame

def save_archor_positive_image():
    POS_PATH = config["POS_PATH"]
    NEG_PATH = config["NEG_PATH"]
    ANC_PATH = config["ANC_PATH"]


    if not os.path.exists(POS_PATH):
        os.makedirs(POS_PATH)
    if not os.path.exists(NEG_PATH):
        os.makedirs(NEG_PATH)
    if not os.path.exists(ANC_PATH):
        os.makedirs(ANC_PATH)

    name = input("Enter your name: ")

    POS_PATH_NAME = os.path.join(POS_PATH, name)
    ANC_PATH_NAME = os.path.join(ANC_PATH, name)

    if not os.path.exists(POS_PATH_NAME):
        os.makedirs(POS_PATH_NAME)
    if not os.path.exists(ANC_PATH_NAME):
        os.makedirs(ANC_PATH_NAME)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        frame = frame[120:120+250, 200:200+250]

        if cv2.waitKey(1) & 0XFF == ord('a'):
            imgname = os.path.join(ANC_PATH_NAME, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(imgname, frame)

        
        if cv2.waitKey(1) & 0XFF == ord('p'):
            imgname = os.path.join(POS_PATH_NAME, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(imgname, frame)

        cv2.imshow('Image Collection', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 

    plt.imshow(frame)
    plt.show()

    print(frame.shape)