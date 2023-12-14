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


def mse(y: ndarray, y_pred: ndarray) -> float:
    """
    Mean square error
    """
    return np.power((y - y_pred), 2).mean()


def norm(a: ndarray, b: ndarray) -> ndarray:
    """
    Norm 2
    """
    return np.linalg.norm((a - b))


def max_min_eigenvalue(H: ndarray, lambda_: float) -> (float, float):
    """
    Compute the hessian of the matrix H and return the maximum and minimum eigenvalues
    """
    hessian = 2 * (H @ H.T + np.power(lambda_, 2) * np.eye(H.shape[0]))
    eigenvalues = np.linalg.eigvals(hessian)
    return np.max(eigenvalues), np.min(eigenvalues)


def get_residual_y(y: ndarray, y_pred: ndarray) -> float:
    return np.linalg.norm(y - y_pred) / np.linalg.norm(y)
