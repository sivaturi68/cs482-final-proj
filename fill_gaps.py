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

# Load videos into memory
cap = cv.VideoCapture('soccer 200/ezgif-frame-%03d.jpg')

frame_no = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print(f'can\'t read frame {frame_no}, ending capture')
        break

    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', grey)
    time.sleep(0.1)

    if cv.waitKey(1) == ord('q'):
        break

    frame += 1

cap.release()
cv.destroyAllWindows()
