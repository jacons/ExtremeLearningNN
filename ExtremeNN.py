from numpy import ndarray, eye
from numpy.linalg import inv
from numpy.random import rand

from NumericalUtils import cholesky, backwardSub, forwardSub


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

        self.resevoir = lambda x: activation(self.w1 @ x.T).T
        self.alfa_reg = alfa_reg  # L2 regularization

    def fit_normal_equation(self, x: ndarray, y: ndarray):
        """
        Fit the neural network using the normal equation
        Find min_w(wH-Y) indeed w = (H.T * H + alpha*I) * H.T * Y
        :param x: Input dataset
        :param y: Target
        :return: None
        """
        h = self.resevoir(x)
        self.w2 = (inv(h.T @ h + self.alfa_reg * eye(h.shape[1])) @ h.T @ y).T
        return

    def fit_cholesky(self, x: ndarray, y: ndarray):
        """
        Fit the neural network using the Cholesky factorization
        In order to find the best "w" we need to solve the following: w(R.T * R) = Y.T * H
        :param x: Input dataset
        :param y: Target
        :return: None
        """

        # Perform the first (resevoir) layer
        h = self.resevoir(x)
        # (1) Apply the cholesky factorization
        r = cholesky(h.T @ h + self.alfa_reg * eye(h.shape[1])).T
        # (2) back substitution (r upper triangular)
        z = backwardSub(r, y.T @ h)
        # (3) forward substitution (r.T lower triangular)
        self.w2 = forwardSub(r.T, z)
        return

    def __call__(self, x: ndarray) -> ndarray:
        hidden = self.resevoir(x)
        return (self.w2 @ hidden.T).T
