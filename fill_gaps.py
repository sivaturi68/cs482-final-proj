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
import deriv

from RANSAC import CubicRANSACModel, QuadraticRANSACModel, LinearRANSACModel

# Load tracking data
tracking = pd.read_csv(
    './TestOut/exp2/detection_results.csv',
    # './TestOut/exp4/detection_results.csv',
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

def get_motion_field(frames):
    frames_gray = [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in frames[-5:]]
    frames_gray = np.array(frames_gray)
    frames_gray = (frames_gray - frames_gray.min()) / (frames_gray.max() - frames_gray.min())

    u_field = np.zeros_like(frames_gray[0]).astype('float32')
    v_field = np.zeros_like(frames_gray[0]).astype('float32')

    I_x = deriv.horiz_deriv(frames_gray).astype('float32')
    I_y = deriv.vert_deriv(frames_gray).astype('float32')
    I_t = deriv.time_deriv(frames_gray).astype('float32')

    l = 0.01

    # while True:
    for i in range(5):
        calculation = (I_x[0] * u_field + I_y[0] * v_field + I_t[0]) / ((1 / l) + I_x[0] ** 2 + I_y[0] ** 2)
        u_field -= calculation * I_x[0]
        v_field -= calculation * I_y[0]

    return u_field, v_field

def motion_field_to_frame(u_field, v_field):
    return np.dstack((u_field, v_field, np.zeros_like(u_field)))

def get_hom_between_frames(frame1, frame2):
    orb = cv.ORB_create(250)

    kp1, ds1 = orb.detectAndCompute(frame1, None)
    kp2, ds2 = orb.detectAndCompute(frame2, None)

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = list(matcher.match(ds1, ds2))
    matches.sort(key=lambda x: x.distance)
    matches = matches[:len(matches) // 2]

    p1 = np.zeros((len(matches), 2))
    p2 = np.zeros((len(matches), 2))

    p1 = np.array([kp1[matches[i].queryIdx].pt for i in range(len(matches))])
    p2 = np.array([kp2[matches[i].trainIdx].pt for i in range(len(matches))])

    H, _ = cv.findHomography(p1, p2, cv.RANSAC)

    return H

def apply_homography(H, x, y):
    Xp = H @ np.array([x, y, 1]).T
    return np.array(Xp[:2]) / Xp[2]


def process_frame(frame):
    global frames, frame_no, vwidth, vheight, data
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

    H = get_hom_between_frames(frames[-1], frames[-2])

    data = [[t, *apply_homography(npl.inv(H), x, y)] for (t, x, y) in data]

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
    # return motion_field_to_frame(*motion_field)
    # return frame

# Load videos into memory
# cap = cv.VideoCapture('soccer_video_static/fk.mp4')
cap = cv.VideoCapture('soccer_video/Soccer_test.mp4')

vwidth  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
vheight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# vwidth  = 1280
# vheight = 720
framerate = float(cap.get(cv.CAP_PROP_FPS))
previous_frame_time = 0

fourcc = cv.VideoWriter_fourcc(*'DIVX')
# out = cv.VideoWriter('soccer_video_static_annotated.avi', fourcc, framerate, (vwidth, vheight))
out = cv.VideoWriter('soccer_video_annotated.avi', fourcc, framerate, (vwidth, vheight))

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

    out.write(annotated)

    if cv.waitKey(1) == ord('q'):
        break

    frame_no += 1

cap.release()
out.release()
cv.destroyAllWindows()
