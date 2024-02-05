import datetime
import sys
from typing import Literal

import numpy as np
from numpy import ndarray, eye
from numpy.linalg import norm

from NumericalUtils import cholesky, backwardSub, forwardSub
from Utils import max_min_eigenvalue, sigmoid, ReLU, tanH


class ENeuralN:

    def __init__(self, hidden: int, regularization: float, resevoir: np.ndarray = None, features: int = 10,
                 activation: Literal["sig", "relu", "tanH"] = "sig"):
        """
        Initialize an instance of the Extreme Neural Network. The Extreme Neural Network is a reservoir computing
        model with an optional activation function. It consists of a resevoir layer (`w1`), readout layer (`w2`),
        and an activation function applied to the resevoir.

        Parameters:
         * hidden (int): Number of hidden nodes in the network.
         * regularization (float): L2 regularization parameter (alpha).
         * resevoir (np.ndarray): Reservoir layer (random if not provided).
         * features (int): Number of input features.
         * activation (Literal["sig", "relu", "tanH"]): Activation function to be used ("sig" for sigmoid, "relu" for
                ReLU, "tanH" for hyperbolic tangent).
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
        Fit the neural network using the Cholesky factorization. This method applies the Cholesky factorization to
        solve for the optimal weights (w2) using the resevoir layer (w1).

        To find the best "w", solve the equation: w(R.T * R) = Y * H.T

        Parameters:
         * x (ndarray): Input dataset.
         * y (ndarray): Target.
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
        """
        Calculate the lambda parameter for FISTA optimization. This function computes the lambda parameter used in
        FISTA optimization based on the input lambda.

        Parameters:
         * lambda_ (float): Input lambda parameter (a time k).

        Returns:
         * float: Calculated lambda parameter for FISTA optimization (a time k+1).

        """
        return (1 + np.sqrt(1 + 4 * np.power(lambda_, 2))) / 2

    def fit_fista(self, x: ndarray, y: ndarray, max_iter: int, eps: float = 0):
        """
        Fit the neural network using FISTA optimization. This method applies FISTA optimization to update the weights
        of the neural network. It uses a fixed step-size and performs iterations until convergence or reaching
        the maximum number of iterations.

        Parameters:
         * x (ndarray): Input dataset.
         * y (ndarray): Target.
         * max_iter (int): Maximum number of iterations.
         * eps (float): Gradient threshold for convergence (default is 0).

        Returns:
         * Tuple: List of weights for each iteration and the final norm of the gradient.
        """
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

        if norm_grad < eps:
            print(f"Converged in {current_iter} iterations. Norm grad: {norm_grad}")

        return weights, norm_grad

    def fit_GD(self, x: ndarray, y: ndarray, max_iter: int,
               lr: float = 0, beta: float = 0, eps: float = 0, testing: bool = False):
        """
        Fit the neural network using Gradient Descent (GD) optimization. This method applies Gradient Descent
        optimization to update the weights of the neural network. It uses an adaptive learning rate if lr is not
        provided. If testing is True, additional information is printed during the training process.

        Parameters:
         * x (ndarray): Input dataset.
         * y (ndarray): Target.
         * max_iter (int): Maximum number of iterations.
         * lr (float): Learning rate for the gradient descent (default is 0, adaptive if lr <= 0).
         * beta (float): Momentum term for the gradient descent (default is 0).
         * eps (float): Gradient threshold for convergence (default is 0).
         * testing (bool): If True, print additional information during training (default is False).

        Returns:
         * Tuple: List of weights for each iteration and the final norm of the gradient.
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
