from numpy import ndarray, eye
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

    def __call__(self, x: ndarray) -> ndarray:
        return self.w2 @ self.resevoir(x)