import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import scipy
import random
import matplotlib.pyplot as plt

class LinearRANSACModel:
    def __init__(self, k=50, n=5):
        self.X = None
        self.y = None

        self.k = k
        self.n = n
        self.coefficients_ = None

    def get_system_mat(self, xs):
        X = np.hstack((xs, np.ones((xs.shape[0], 1))))
        return X

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        self.n = min(self.n, xs.shape[0] - 1)

        best_coeffs = None
        best_score = np.inf

        for _ in range(self.k):
            rand_rows = npr.randint(xs.shape[0], size=(self.n))
            coeffs = self.solve(xs[rand_rows], ys[rand_rows])

            X = self.get_system_mat(xs[rand_rows])
            predicted = X @ coeffs

            score = np.abs((predicted ** 2 - ys[rand_rows] ** 2).sum())

            if score < best_score:
                best_score = score
                best_coeffs = coeffs

        self.coefficients_ = best_coeffs

    def transform(self, xs):
        X = self.get_system_mat(xs)
        return X @ self.coefficients_

    def solve(self, xs, ys):
        X = self.get_system_mat(xs)
        coeffs, residuals, rank, s = npl.lstsq(X, ys)
        return coeffs

class QuadraticRANSACModel(LinearRANSACModel):
    def get_system_mat(self, xs):
        X = np.hstack((xs ** 2, xs, np.ones((xs.shape[0], 1))))
        return X
class CubicRANSACModel(LinearRANSACModel):
    def get_system_mat(self, xs):
        X = np.hstack((xs ** 3, xs ** 2, xs, np.ones((xs.shape[0], 1))))
        return X

if __name__ == '__main__':
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
            ransac = QuadraticRANSACModel(k=50, n=5)

            x_dims = 1
            num_x = 10

            actual_coeffs = npr.random((x_dims + 2, 1))
            xs = np.linspace(0, 10, num_x)[:, np.newaxis]
            actual_ys = (ransac.get_system_mat(xs) @ actual_coeffs) * (npr.random((num_x, 1)) * 0.1 + .9)

            ransac.fit(xs, actual_ys)
            print(f'coefficients: {ransac.coefficients_}')

            predicted_ys = (ransac.get_system_mat(xs) @ ransac.coefficients_)

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

