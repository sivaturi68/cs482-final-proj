import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import scipy
import random
import matplotlib.pyplot as plt

class LinearRANSACModel:
    def __init__(self, k=50, n=5, last=10):
        self.X = None
        self.y = None

        # number of iterations
        self.k = k
        # number of points to least-squares on
        self.n = n
        self.coefficients_ = None
        self.last = last

    def get_system_mat(self, xs):
        # put 1 at the end of every vector to have a "bias" term
        X = np.hstack((xs, np.ones((xs.shape[0], 1))))
        return X

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        # make sure we dont over-sample points
        self.n = min(self.n, xs.shape[0] - 1)

        # keep track of best so far
        best_coeffs = None
        best_score = np.inf

        for _ in range(self.k):
            # pick random points and solve for the least-squares parameters
            low = max(0, xs.shape[0] - self.last)
            high = xs.shape[0]
            rand_rows = npr.randint(low, high, size=(self.n))
            coeffs = self.solve(xs[rand_rows], ys[rand_rows])

            # apply predicted model
            X = self.get_system_mat(xs[rand_rows])
            predicted = X @ coeffs

            # score the model based on number of inliers
            score = self.n
            diffs = np.abs(predicted - ys[rand_rows])
            thresh = .25 * (diffs.max() - diffs.min()) # inliers are within 25% of error range
            score -= len(diffs[diffs < thresh])

            if score < best_score:
                best_score = score
                best_coeffs = coeffs

        self.coefficients_ = best_coeffs

    def transform(self, xs):
        # just a matrix multiplication to apply predicted model
        X = self.get_system_mat(xs)
        return X @ self.coefficients_

    def solve(self, xs, ys):
        # find least-squares parameters
        X = self.get_system_mat(xs)
        coeffs, residuals, rank, s = npl.lstsq(X, ys)
        return coeffs

class QuadraticRANSACModel(LinearRANSACModel):
    # treat the x**2 term as another input dimension
    def get_system_mat(self, xs):
        X = np.hstack((xs ** 2, xs, np.ones((xs.shape[0], 1))))
        return X

class CubicRANSACModel(LinearRANSACModel):
    # treat the x**2 and x**3 terms as other input dimensions
    def get_system_mat(self, xs):
        X = np.hstack((xs ** 3, xs ** 2, xs, np.ones((xs.shape[0], 1))))
        return X

if __name__ == '__main__':
    # rough testing (feel free to run this file)
    method = 'quadratic'
    match method:
        case 'cubic':
            ransac = CubicRANSACModel(k=50, n=5)

            x_dims = 1
            num_x = 10

            actual_coeffs = npr.random((x_dims + 3, 1))
            xs = np.linspace(0, 10, num_x)[:, np.newaxis]
            actual_ys = (ransac.get_system_mat(xs) @ actual_coeffs) * (npr.random((num_x, 1)) * 0.1 + .9)

            ransac.fit(xs, actual_ys)
            print(f'coefficients: {ransac.coefficients_}')

            predicted_ys = (ransac.get_system_mat(xs) @ ransac.coefficients_)

            plt.figure()
            plt.scatter(xs, actual_ys)
            plt.plot(xs, predicted_ys)

            plt.show()
        case 'quadratic':
            ransac = QuadraticRANSACModel(k=50, n=5, last=500)

            x_dims = 1
            num_x = 30

            actual_coeffs = npr.random((x_dims + 2, 1))
            xs = np.linspace(0, 10, num_x)[:, np.newaxis]
            actual_ys = (ransac.get_system_mat(xs) @ actual_coeffs) * (npr.random((num_x, 1)) * 0.1 + .9)

            ransac.fit(xs, actual_ys)
            print(f'coefficients: {ransac.coefficients_}')

            # predicted_ys = (ransac.get_system_mat(xs) @ ransac.coefficients_)
            predicted_ys = ransac.transform(xs)

            plt.figure()
            plt.scatter(xs, actual_ys)
            plt.plot(xs, predicted_ys)

            plt.show()
        case 'linear':
            ransac = LinearRANSACModel(k=50, n=5)

            x_dims = 1
            num_x = 10

            actual_coeffs = npr.random((x_dims + 1, 1))
            xs = np.linspace(0, 10, num_x)[:, np.newaxis]
            actual_ys = (ransac.get_system_mat(xs) @ actual_coeffs) * (npr.random((num_x, 1)) * 0.1 + .9)

            ransac.fit(xs, actual_ys)
            print(f'coefficients: {ransac.coefficients_}')

            predicted_ys = (ransac.get_system_mat(xs) @ ransac.coefficients_)

            plt.figure()
            plt.scatter(xs, actual_ys)
            plt.plot(xs, predicted_ys)

            plt.show()

