import numpy as np
import numpy.linalg as npl
import scipy
import random

# def RANSAC_create(regression_type=None):
#     if regression_type is None:
#         regression_type = 'linear'

#     if regression_type not in ['linear', 'quadratic']:
#         raise Exception(f'regression type must be either linear or quadratic, not {regression_type}')

#     match regression_type:
#         case 'linear':
#             return RANSAC_linear()

class LinearRANSAC:
    def __init__(self, k=50):
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        best_coeffs = None
        best_score = np.inf

        for i in range(self.k):
            rand_X = np.array(random.sample(X, k=2))
            coeffs = self.solve(*rand_X.flatten())

    def solve(self, xs, ys):
        x1, x2 = xs
        y1, y2 = ys

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

