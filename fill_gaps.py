import numpy as np
import numpy.linalg as npl
# import scipy
import pandas as pd
import cv2 as cv
from PIL import Image
# import sklearn
# from sklearnex import patch_sklearn
import matplotlib.pyplot as plt
import bokeh as bk
import time

from RANSAC import CubicRANSACModel, QuadraticRANSACModel, LinearRANSACModel

# Load tracking data
tracking = pd.read_csv(
    './TestOut/exp4/detection_results.csv',
    header=0,
    index_col=0,
    names=['frame_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence'])

def reformat_row(row):
    cid, conf, x, y, w, h = row
    cid = int(cid)
    conf, x, y, w, h = [float(t) for t in (conf, x, y, w, h)]

    return pd.Series([cid, conf, x, y, w, h])

# tracking = tracking.apply(reformat_row, axis=1)

tracking = tracking[~tracking.index.duplicated(keep='first')].astype('float32')

new_indices = {fid:int(fid.split('_')[-1]) for fid in tracking.index}

tracking = tracking.rename(index=new_indices)

frames = []
frame_no = 0

data = []

N = 4
LAST = 5

ransac = QuadraticRANSACModel(k=100, n=N, last=LAST)

# def get_motion_field(frames):
#     u_field = np.zeros_like(frame1)
#     v_field = np.zeros_like(frame1)
#     zeroes = np.zeros_like(frame1)

#     l = 0.01

#     # while True:
#     for i in range(100):
#         calculation = (I_x[0] * u_field + I_y[0] * v_field + I_t[0]) / ((1 / l) + I_x[0] ** 2 + I_y[0] ** 2)
#         u_field -= calculation * I_x[0]
        # v_field -= calculation * I_y[0]

def process_frame(frame):
    global frames, frame_no, vwidth, vheight
    frames.append(frame)

    if frame_no in tracking.index:
        _, x_center, y_center, _, _, conf = tracking.loc[frame_no]
        if conf >= 0.30:
            # t, x, y = frame_no, (x_center + 1) * vwidth, -1 * (y_center - 1) * vheight
            t, x, y = frame_no, x_center * vwidth, y_center * vheight
            data.append([t, x, y])
            frame = cv.circle(frame, np.int32([x, y]), 5, (255, 0, 0), -1)
        else:
            print(f'confident frame #{frame_no}')

    if len(data) < N or len(data) < LAST:
        return frame

    mat = np.array(data)
    ts, xs, ys = mat.T
    sample_ts = np.linspace(frame_no - 30, frame_no + 30, 300)

    ransac.fit(ts[:, np.newaxis], xs[:, np.newaxis])
    predicted_xs = ransac.transform(sample_ts[:, np.newaxis])
    ransac.fit(ts[:, np.newaxis], ys[:, np.newaxis])
    predicted_ys = ransac.transform(sample_ts[:, np.newaxis])

    polyline_arr = np.int32([
        np.array([predicted_xs[:, 0], predicted_ys[:, 0]]).T
        ])

    annotated = cv.polylines(
        frame,
        polyline_arr,
        False, (255, 0, 0))

    return annotated
    # return frame

# Load videos into memory
cap = cv.VideoCapture('soccer_video_static/fk.mp4')
# cap = cv.VideoCapture('./TestOut/exp2/labels/Soccer_test.mp4')

vwidth  = cap.get(cv.CAP_PROP_FRAME_WIDTH)
vheight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
# vwidth  = 1280
# vheight = 720
framerate = cap.get(cv.CAP_PROP_FPS)
previous_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print(f'can\'t read frame {frame_no}, ending capture')
        break

    # grey = cv.resize(frame, (1280, 720))
    grey = frame

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

    frame_no += 1

cap.release()
cv.destroyAllWindows()
