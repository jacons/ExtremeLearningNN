import numpy as np
from numpy import ndarray


def sigmoid(m: ndarray) -> ndarray:
    """
    This function implements the sigmoid activation function. The sigmoid function is commonly used in neural networks
    to introduce non-linearity. It squashes input values between 0 and 1, which is useful for binary classification
    problems or as an activation function in the hidden layers of a neural network.

    Parameters:
     * m (ndarray): Input matrix.

    Returns:
     * ndarray: Output matrix after applying the sigmoid activation.
    """
    return 1 / (1 + np.exp(-m))


def ReLU(m: ndarray) -> ndarray:
    """
    This function implements the Rectified Linear Unit (ReLU) activation function. The ReLU activation function
    introduces non-linearity by setting negative values to zero and leaving positive values unchanged. It is
    commonly used in neural networks, helping them learn complex patterns in the data.

    Parameters:
     * m (ndarray): Input matrix.

    Returns:
     * ndarray: Output matrix after applying the ReLU activation.
    """
    return np.maximum(0, m)


def tanH(m: ndarray) -> ndarray:
    """
    This function implements the hyperbolic tangent (tanh) activation function. The tanh activation function squashes
    the input values to the range of [-1, 1], introducing non-linearity. It is commonly used in neural networks,
    similar to sigmoid, but with a range that includes negative values

    Parameters:
     * m (ndarray): Input matrix.

    Returns:
     * ndarray: Output matrix after applying the tanh activation.

    """
    return np.tanh(m)


def mse(y: ndarray, y_pred: ndarray) -> float:
    """
    This function calculates the Mean Squared Error (MSE) between actual and predicted values. The Mean Squared Error
    is a common metric used to measure the average squared difference between the actual and predicted values.
    It provides a measure of how well a prediction model performs.

    Parameters:
     * y (ndarray): Actual values.
     * y_pred (ndarray): Predicted values.

    Returns:
     * float: Mean Squared Error.
    """
    return np.power((y - y_pred), 2).mean()


def norm(a: ndarray, b: ndarray) -> ndarray:
    """
    This function calculates the Frobenius norm (L2 norm) between two matrices. It is often used to measure the
    magnitude of a matrix.

    Parameters:
     * a (ndarray): First matrix.
     * b (ndarray): Second matrix.

    Returns:
     * float: Frobenius norm between matrices a and b.

    """
    return np.linalg.norm((a - b), ord="fro")


def max_min_eigenvalue(H: ndarray, lambda_: float) -> (float, float):
    """
    This function computes the Hessian matrix of the input matrix H and returns its maximum and minimum eigenvalues.

    Parameters:
     * H (ndarray): Input matrix.
     * lambda_ (float): Regularization term.

    Returns:
     * Tuple[float, float]: Maximum and minimum eigenvalues of the Hessian matrix.

    """
    hessian = 2 * (H @ H.T + np.power(lambda_, 2) * np.eye(H.shape[0]))
    eigenvalues = np.linalg.eigvals(hessian)
    return np.max(eigenvalues), np.min(eigenvalues)


def get_gap_sol(w2: ndarray, w2_star: ndarray) -> float:
    """
    This function calculates the relative Frobenius norm of the difference between two weight matrices.

    Parameters:
     * w2 (ndarray): The computed weight matrix.
     * w2_star (ndarray): The reference or optimal weight matrix.

    Returns:
     * float: The relative Frobenius norm of the difference between w2 and w2_star.
    """
    return np.linalg.norm(w2 - w2_star, ord="fro") / np.linalg.norm(w2_star, ord="fro")
