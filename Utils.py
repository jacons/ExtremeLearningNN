import numpy as np
from numpy import ndarray


def sigmoid(m: ndarray) -> ndarray:
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-m))


def ReLU(m: ndarray) -> ndarray:
    """
    ReLU activation functon
    """
    return np.maximum(0, m)


def tanH(m: ndarray) -> ndarray:
    """
    Hyperbolic tangent activation function
    """
    return np.tanh(m)


def MSE(y: ndarray, y_pred: ndarray) -> float:
    """
    Mean square error
    """
    return np.power((y - y_pred), 2).mean()


def norm(a: ndarray, b: ndarray) -> ndarray:
    """
    Norm 2
    """
    return np.linalg.norm((a - b))
