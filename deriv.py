import scipy
import numpy as np

convolve = lambda img, flt: scipy.ndimage.convolve(img, flt, mode='nearest')

def horiz_deriv(video: np.ndarray):
    filt = np.array([
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        [[-2, 0, 2],
         [-4, 0, 4],
         [-2, 0, 2]],
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1],]
    ])

    return convolve(video, filt)

def vert_deriv(video: np.ndarray):
    filt = np.array([
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        [[-2, -4, -2],
         [0, 0, 0],
         [2, 4, 2]],
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
    ])

    return convolve(video, filt)

def time_deriv(video: np.ndarray):
    filt = np.array([
        [[-1, -2, -1],
         [-2, -4, -2],
         [-1, -2, -1]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]],
    ])

    return convolve(video, filt)