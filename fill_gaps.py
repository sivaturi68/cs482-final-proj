import numpy as np
import numpy.linalg as npl
import scipy
import pandas as pd
import cv2 as cv
from PIL import Image
import sklearn
from sklearnex import patch_sklearn
import matplotlib.pyplot as plt
import bokeh as bk
import time

from RANSAC import QuadraticRANSACModel

# Load tracking data
tracking = pd.read_csv(
    './TestOut/exp2/detection_results.csv',
    header=0,
    names=['frame_id', 'class_id', 'confidence', 'x_center', 'y_center', 'width', 'height'])

frames = []
frame_no = 0

def process_frame(frame):
    global frames, frame_no
    frames.append(frame)

    frame_no += 1
    return frame

# Load videos into memory
cap = cv.VideoCapture('soccer_video/Soccer_test.mp4')

framerate = 30
previous_frame_time = 0

frame_no = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print(f'can\'t read frame {frame_no}, ending capture')
        break

    grey = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (1280, 720))

    delta = time.time() - previous_frame_time
    while delta < 1 / framerate:
        # continue
        time.sleep(0.01)
        delta = time.time() - previous_frame_time
    previous_frame_time = time.time()

    annotated = process_frame(grey)
    cv.imshow('frame', annotated)

    if cv.waitKey(1) == ord('q'):
        break

    frame += 1

cap.release()
cv.destroyAllWindows()
