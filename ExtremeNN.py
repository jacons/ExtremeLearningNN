import numpy as np
from numpy import ndarray, eye
from numpy.random import rand

from NumericalUtils import cholesky, backwardSub, forwardSub
from Utils import max_eigenvalue, MSE


class ENeuralN:

    def __init__(self, features: int, hidden: int, regularization: float, activation):
        """
        Implementation of Extreme Neural network
        :param features: Number of input features
        :param hidden: Hidden nodes
        :param regularization: L2 regularization alpha
        :param activation: activation function
        """
        self.w1 = rand(hidden, features)  # resevoir
        self.w2 = None  # readout

        self.resevoir = lambda x: activation(self.w1 @ x)
        self.regularization = regularization  # L2 regularization

    def fit_cholesky(self, x: ndarray, y: ndarray):
        """
        Fit the neural network using the Cholesky factorization
        to find the best "w" we need to solve the following: w(R.T * R) = Y * H.T
        :param x: Input dataset
        :param y: Target
        :return: None
        """
        if x.shape[1] < self.w1.shape[0]:
            print("Error")
            return
        # Perform the first (resevoir) layer
        h = self.resevoir(x)
        # (1) Apply the cholesky factorization
        r = cholesky(h @ h.T + self.regularization * eye(h.shape[0])).T
        # (2) back substitution (r upper triangular)
        z = backwardSub(r, y @ h.T)
        # (3) forward substitution (r.T lower triangular)
        self.w2 = forwardSub(r.T, z)
        return

    @staticmethod
    def calc_lambda(lambda_):
        return (1 + np.sqrt(1 + 4 * np.power(lambda_, 2))) / 2

    def fit_fista(self, x: ndarray, y: ndarray, max_iter: int) -> list[float]:
        # Perform the first (resevoir) layer
        h = self.resevoir(x)
        # Perform the lipschitz constant
        L = max_eigenvalue(h)

        # Initialize with random matrix (random weight)
        self.w2 = rand(2, h.shape[0])
        # "Previous weight"
        w2_old = self.w2.copy()

        c, lambda_k_1, current_iter = 0, 0, 0
        mse_errors = []

        while current_iter < max_iter:
            z_k = w2_old + c * (self.w2 - w2_old)
            grad_zk = (z_k @ h - y) @ h.T
            self.w2 = z_k - (1 / L) * grad_zk - self.regularization * self.w2

            lambda_k = self.calc_lambda(lambda_k_1)
            c = (lambda_k_1 - 1) / lambda_k
            lambda_k_1 = lambda_k
            w2_old = self.w2.copy()

            y_pred = self.w2 @ h
            mse_error = MSE(y, y_pred)
            mse_errors.append(mse_error)
            current_iter += 1

        return mse_errors

    def fit_standard_SDG(self, x: ndarray, y: ndarray, max_iter: int):
        # TO DO
        pass

    def __call__(self, x: ndarray) -> ndarray:
        return self.w2 @ self.resevoir(x)


"""
        c0 = 1
        y_0 = self.w2 * c0 * (self.w2 - w2_old)
        grad_y_0 = (y_0 @ h - target) @ h.T
        self.w2 = y_0 - (1 / L) * grad_y_0

        c1 = (lambda_2 - 1) / lambda_1
        y_1 = self.w2 + c1 * (self.w2 - w2_old)
        grad_y_1 = (y_0 @ h - target) @ h.T
        self.w2 = y_1 - (1 / L) * grad_y_1
"""
