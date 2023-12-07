import sys

import numpy as np
from numpy import ndarray, eye
from numpy.linalg import norm
from numpy.random import rand

from NumericalUtils import cholesky, backwardSub, forwardSub
from Utils import maxmin_eigenvalue, MSE


class ENeuralN:

    def __init__(self, features: int, hidden: int, regularization: float, activation,
                 resevoir: np.ndarray):
        """
        Implementation of Extreme Neural network
        :param features: Number of input features
        :param hidden: Hidden nodes
        :param regularization: L2 regularization alpha
        :param activation: activation function
        """
        if resevoir is None:
            self.w1 = np.random.uniform(-1, 1, (hidden, features))  # resevoir
        else:
            self.w1 = resevoir.copy()

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

    def fit_fista(self, x: ndarray, y: ndarray, max_iter: int, eps: float = 0, w_star=None):
        """
        :param x: array X [ feature, examples ]
        :param y: array target [ 2, examples ]
        :param max_iter: Number of max iteration
        :param eps: gradient threshold
        :return:
        """
        # Perform the first (resevoir) layer
        h = self.resevoir(x)

        # Perform the lipschitz constant
        L, tau = maxmin_eigenvalue(h)

        # Initialize with random matrix (random weight)
        self.w2 = np.random.uniform(-1, 1, (2, h.shape[0]))
        # "Previous weight"
        w2_old = self.w2.copy()

        beta, lambda_k_1, current_iter = 0, 0, 0
        step_size = 2 / (L + tau)  # step-size changed from 1/L
        mse_errors = []
        gaps = []
        grad_zk = sys.maxsize

        while (current_iter < max_iter) and (norm(grad_zk) > eps):
            # ---- Gradient ----
            z_k = w2_old + beta * (self.w2 - w2_old)
            grad_zk = (z_k @ h - y) @ h.T
            # ---- Gradient ----

            # ---- Update rule ----
            self.w2 = z_k - step_size * grad_zk - self.regularization * self.w2
            # ---- Update rule ----

            # ---- Update "beta" ----
            lambda_k = self.calc_lambda(lambda_k_1)
            beta = (lambda_k_1 - 1) / lambda_k
            lambda_k_1 = lambda_k
            # ---- Update "beta" ----

            w2_old = self.w2.copy()

            # output predicted
            y_pred = self.w2 @ h
            mse_error = MSE(y, y_pred)  # MSE between the target and the output predicted
            mse_errors.append(mse_error)
            gaps.append(np.linalg.norm(self.w2 - w_star) / np.linalg.norm(w_star))
            current_iter += 1

        return mse_errors, gaps

    def fit_SDG(self, x: ndarray, y: ndarray, max_iter: int,
                lr: float, beta: float = 0, eps: float = 0, w_star: np.ndarray = None):
        """
        :param w_star:
        :param x: array X [ feature, examples ]
        :param y: array target [ 2, examples ]
        :param max_iter: Number of max iteration
        :param lr: learning rate if 0 then will be used 1/L
        :param beta: momentum term
        :param eps: gradient threshold
        """
        # Perform the first (resevoir) layer
        h = self.resevoir(x)

        L, tau = maxmin_eigenvalue(h)
        lr = 1 / (L + tau) if lr <= 0 else lr

        # Initialize with random matrix (random weight)
        self.w2 = np.random.uniform(-1, 1, (2, h.shape[0]))

        # "Previous weight"
        w2_old = self.w2.copy()
        w2_old_old = self.w2.copy()

        mse_errors = []
        gaps = []
        grad_w2 = sys.maxsize
        current_iter = 0

        while (current_iter < max_iter) and (norm(grad_w2) > eps):
            grad_w2 = (w2_old @ h - y) @ h.T
            # ---- Update rule ----
            self.w2 = w2_old - lr * grad_w2 + beta * (w2_old - w2_old_old) - self.regularization * self.w2
            # ---- Update rule ----

            w2_old_old = w2_old.copy()
            w2_old = self.w2.copy()

            # output predicted
            y_pred = self.w2 @ h
            mse_error = MSE(y, y_pred)  # MSE between the target and the output predicted
            mse_errors.append(mse_error)
            gaps.append(np.linalg.norm(self.w2 - w_star) / np.linalg.norm(w_star))
            current_iter += 1
        return mse_errors, gaps

    def __call__(self, x: ndarray) -> ndarray:
        return self.w2 @ self.resevoir(x)
