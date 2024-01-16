import sys
from typing import Literal

import numpy as np
from numpy import ndarray, eye
from numpy.linalg import norm

from NumericalUtils import cholesky, backwardSub, forwardSub
from Utils import max_min_eigenvalue, mse, sigmoid, ReLU, tanH

import datetime


class ENeuralN:

    def __init__(self, hidden: int, regularization: float, resevoir: np.ndarray = None, features: int = 10,
                 activation: Literal["sig", "relu", "tanH"] = "sig"):
        """
        Implementation of Extreme Neural network
        :param features: Number of input features
        :param hidden: Hidden nodes
        :param regularization: L2 regularization alpha
        :param activation: activation function
        """
        # -------- Set the resevoir --------
        # If it is not provided will be generated a random one
        if resevoir is None:
            self.w1 = np.random.uniform(-1, 1, (hidden, features))  # resevoir
        else:
            self.w1 = resevoir.copy()
        # -------- Set the resevoir --------

        # -------- Set the activation function --------
        if activation == "sig":
            activation = sigmoid
        elif activation == "relu":
            activation = ReLU
        else:
            activation = tanH
        # -------- Set the activation function --------

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
        r = cholesky(h @ h.T + np.power(self.regularization, 2) * eye(h.shape[0])).T
        # (2) back substitution (r upper triangular)
        z = backwardSub(r, y @ h.T)
        # (3) forward substitution (r.T lower triangular)
        self.w2 = forwardSub(r.T, z)
        return

    @staticmethod
    def calc_lambda(lambda_):
        return (1 + np.sqrt(1 + 4 * np.power(lambda_, 2))) / 2

    def fit_fista2(self, x: ndarray, y: ndarray, max_iter: int, eps: float = 0):

        # Perform the first (resevoir) layer
        h = self.resevoir(x)

        # Perform the lipschitz constant
        L, tau = max_min_eigenvalue(H=h, lambda_=self.regularization)

        # Initialize with random matrix (random weight)
        self.w2 = np.random.uniform(-1, 1, (2, h.shape[0]))
        # "Previous weight"
        w2_old, z = self.w2.copy(), self.w2.copy()

        # fixed step-size
        step_size = 1 / (L + tau)

        # We have:gamma_k_1 (lambda k+1), current_iter (Current iteration)
        lambda_k_1, current_iter = 0, 0

        # List of w, one for each iteration
        weights = []

        norm_grad = sys.maxsize

        def grad(c):
            return 2 * ((c @ (h @ h.T)) - (y @ h.T) + (np.power(self.regularization, 2) * c))

        while (current_iter < max_iter) and (norm_grad > eps):
            grad_z = grad(z)
            self.w2 = z - (step_size * grad_z)

            lambda_k = self.calc_lambda(lambda_k_1)
            beta = (lambda_k_1 - 1) / lambda_k
            z = self.w2 + (beta * (self.w2 - w2_old))
            lambda_k_1 = lambda_k

            w2_old = self.w2.copy()

            weights.append(self.w2.copy())
            current_iter += 1
            norm_grad = norm(grad_z)

            # print(f"norm_grad: {norm_grad}")

        if norm_grad < eps:
            print(f"Converged in {current_iter} iterations. Norm grad: {norm_grad}")

        return weights, norm_grad

    def fit_SDG(self, x: ndarray, y: ndarray, max_iter: int,
                lr: float = 0, beta: float = 0, eps: float = 0, testing: bool = False):
        """
        :param x: array X [ feature, examples ]
        :param y: array target [ 2, examples ]
        :param max_iter: Number of max iteration
        :param lr: learning rate if 0 then will be used 1/L
        :param beta: momentum term
        :param eps: gradient threshold
        :param testing: set to true only if you are testing SGD with many iterations,
        if true information about elapsed time and norm grad will be printed
        """

        start = datetime.datetime.now()

        # Perform the first (resevoir) layer
        h = self.resevoir(x)
        # Perform the lipschitz constant
        L, tau = max_min_eigenvalue(h, self.regularization)
        # Initialize with random matrix (random weight)
        self.w2 = np.random.uniform(-1, 1, (2, h.shape[0]))
        # "Previous weight"
        w2_old = self.w2.copy()

        # current_iter (Current iteration)
        current_iter = 0

        # fixed step-size
        lr = 1 / (L + tau) if lr <= 0 else lr
        # List of w, one for each iteration
        weights = []

        norm_grad = sys.maxsize

        def grad(c):
            return 2 * ((c @ (h @ h.T)) - (y @ h.T) + (np.power(self.regularization, 2) * c))

        it = 0

        while (current_iter < max_iter) and (norm_grad > eps):
            grad_w2 = grad(self.w2)
            # ---- Update rule ----
            self.w2 = self.w2 - (lr * grad_w2) + (beta * (self.w2 - w2_old))
            w2_old = self.w2.copy()
            # ---- Update rule ----

            weights.append(self.w2.copy())
            current_iter += 1
            norm_grad = norm(grad_w2)

            if testing:
                if it % 5000 == 0:
                    elapsed_time = datetime.datetime.now() - start
                    elapsed_seconds = elapsed_time.total_seconds()

                    # Extract seconds and milliseconds
                    seconds = int(elapsed_seconds)
                    milliseconds = int((elapsed_seconds - seconds) * 1000)

                    print(f"step: {it} norm_grad: {norm_grad} time: {seconds}.{milliseconds} (seconds)")

            it += 1

        if norm_grad < eps:
            print(f"Converged in {current_iter} iterations. Norm grad: {norm_grad}")

        if testing:
            elapsed_time = datetime.datetime.now() - start
            elapsed_seconds = elapsed_time.total_seconds()

            # Extract seconds and milliseconds
            seconds = int(elapsed_seconds)
            milliseconds = int((elapsed_seconds - seconds) * 1000)

            print(f"step: {it} norm_grad: {norm_grad} time: {seconds}.{milliseconds} (seconds)")

        return weights, norm_grad

    def __call__(self, x: ndarray) -> ndarray:
        return self.w2 @ self.resevoir(x)
