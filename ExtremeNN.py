import random

import numpy as np
from numpy import ndarray, eye
from numpy.random import rand

from NumericalUtils import cholesky, backwardSub, forwardSub
from Utils import max_eigenvalue, MSE

import matplotlib.pyplot as plt


class ENeuralN:

    def __init__(self, features: int, hidden: int, alfa_reg: float, activation):
        """
        Implementation of Extreme Neural network
        :param features: Number of input features
        :param hidden: Hidden nodes
        :param alfa_reg: L2 regularization alpha
        :param activation: activation function
        """
        self.w1 = rand(hidden, features)  # resevoir
        self.w2 = None  # readout

        self.resevoir = lambda x: activation(self.w1 @ x)
        self.alfa_reg = alfa_reg  # L2 regularization

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
        r = cholesky(h @ h.T + self.alfa_reg * eye(h.shape[0])).T
        # (2) back substitution (r upper triangular)
        z = backwardSub(r, y @ h.T)
        # (3) forward substitution (r.T lower triangular)
        self.w2 = forwardSub(r.T, z)
        return

    @staticmethod
    def calc_lambda(lambda_):
        return (1 + np.sqrt(1 + 4 * np.power(lambda_, 2))) / 2

    def fit_fista(self, x: ndarray, target: ndarray, iter_: int):
        # Perform the first (resevoir) layer
        h = self.resevoir(x)
        L = max_eigenvalue(h)
        print(1/L)
        self.w2 = rand(2, h.shape[0])
        w2_old = self.w2.copy()
        c = 0
        lambda_k_1 = 0

        learning = []
        for k in range(iter_):
            y_k = w2_old + c * (self.w2 - w2_old)
            grad_yk = (y_k @ h - target) @ h.T
            self.w2 = y_k - (1/L) * grad_yk - self.alfa_reg * self.w2

            lambda_k = self.calc_lambda(lambda_k_1)
            c = (lambda_k_1 - 1) / lambda_k
            lambda_k_1 = lambda_k
            w2_old = self.w2.copy()

            y_pred = self.w2 @ h

            mse = MSE(target, y_pred)
            learning.append(mse)
            print(mse)

        plt.plot(learning)
        plt.ylim([-0.5, 10])
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

    def __call__(self, x: ndarray) -> ndarray:
        return self.w2 @ self.resevoir(x)
