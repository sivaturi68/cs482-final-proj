import numpy as np
import numpy.linalg as npl
import pandas as pd
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import bokeh as bk
import time
import deriv

from RANSAC import CubicRANSACModel, QuadraticRANSACModel, LinearRANSACModel

#region Load tracking data
tracking = pd.read_csv(
    './TestOut/exp2/detection_results.csv',
    # './TestOut/exp4/detection_results.csv',
    header=0,
    index_col=0,
    names=['frame_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence'])

# reindex dataframe
tracking = tracking[~tracking.index.duplicated(keep='first')].astype('float32')
new_indices = {fid:int(fid.split('_')[-1]) for fid in tracking.index}
tracking = tracking.rename(index=new_indices)
#endregion

#region globals
frames = []
frame_no = 0
data = []
N = 4
LAST = 5
#endregion

#region trajectory prediction

# custom RANSAC for regressing on history of points
# ransac = QuadraticRANSACModel(k=100, n=N, last=LAST)
ransac = LinearRANSACModel(k=100, n=N, last=LAST)

# find transformation between two frames
def get_hom_between_frames(frame1, frame2):
    # get features
    orb = cv.ORB_create(250)
    kp1, ds1 = orb.detectAndCompute(frame1, None)
    kp2, ds2 = orb.detectAndCompute(frame2, None)

    # find matches
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = list(matcher.match(ds1, ds2))
    matches.sort(key=lambda x: x.distance)
    matches = matches[:len(matches) // 2]

    # calculate homography
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

    # if we have a solid track, add it to the list of known points
    if frame_no in tracking.index:
        _, x_center, y_center, _, _, conf = tracking.loc[frame_no]
        if conf >= 0.30:
            t, x, y = frame_no, x_center * vwidth, y_center * vheight
            data.append([t, x, y])

            # draw dot on the tracked ball
            frame = cv.circle(frame, np.int32([x, y]), 5, (255, 0, 0), -1)
        else:
            print(f'confident frame #{frame_no}')

    # wait until we have enough data to start predicting
    if len(data) < N or len(data) < LAST:
        return frame

    # correct for camera motion
    H = get_hom_between_frames(frames[-1], frames[-2])
    data = [[t, *apply_homography(npl.inv(H), x, y)] for (t, x, y) in data]

    mat = np.array(data)
    ts, xs, ys = mat.T
    sample_ts = np.linspace(frame_no - 30, frame_no + 30, 300)

    # solve for function parameters
    ransac.fit(ts[:, np.newaxis], xs[:, np.newaxis])
    predicted_xs = ransac.transform(sample_ts[:, np.newaxis])
    ransac.fit(ts[:, np.newaxis], ys[:, np.newaxis])
    predicted_ys = ransac.transform(sample_ts[:, np.newaxis])

    # draw line on frame
    polyline_arr = np.int32([
        np.array([predicted_xs[:, 0], predicted_ys[:, 0]]).T
        ])

    annotated = cv.polylines(
        frame,
        polyline_arr,
        False, (255, 0, 0))

    return annotated
#endregion

#region main
# Load videos into memory

# cap = cv.VideoCapture('soccer_video_static/fk.mp4')
cap = cv.VideoCapture('soccer_video/Soccer_test.mp4')

# get video info
vwidth  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
vheight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
framerate = float(cap.get(cv.CAP_PROP_FPS))
previous_frame_time = 0

# prepare to write annotated video
fourcc = cv.VideoWriter_fourcc(*'DIVX')
# out = cv.VideoWriter('soccer_video_static_annotated.avi', fourcc, framerate, (vwidth, vheight))
out = cv.VideoWriter('soccer_video_annotated.avi', fourcc, framerate, (vwidth, vheight))

while cap.isOpened():
    # read and process every frame
    ret, frame = cap.read()

    if not ret:
        print(f'can\'t read frame {frame_no}, ending capture')
        break

    # try not to go faster than framerate
    delta = time.time() - previous_frame_time
    while delta < 1 / framerate:
        time.sleep(0.01)
        delta = time.time() - previous_frame_time
    previous_frame_time = time.time()

    # annotate video with trajectory prediction
    annotated = process_frame(frame)
    cv.imshow('frame', annotated)
    out.write(annotated)

    if cv.waitKey(1) == ord('q'):
        break

    frame_no += 1

cap.release()
out.release()
cv.destroyAllWindows()
#endregion
