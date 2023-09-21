import numpy as np
from numpy import ndarray, zeros_like, sqrt


def cholesky(matrix: ndarray) -> ndarray:
    """
    This function implements the Cholesky decomposition in numpy.
    The matrix is positive definitive can be decomposed in A = "R' x R" where
    R is "upper triangular". Complexity is equal to O((n^3)/3)
    :param matrix: Positive definitive square matrix
    :return: R upper triangular
    """
    r = zeros_like(matrix)  # Upper triangula matrix
    n = matrix.shape[0]  # matrix dimension

    for j in range(n):
        for i in range(j, n):
            if i == j:
                r[i, j] = sqrt(matrix[i, j] - np.sum(r[i, :] ** 2))
            else:
                r[i, j] = (matrix[i, j] - np.sum(r[i, :] * r[j, :])) / r[j, j]

    return r  # Upper triangula matrix


def backwardSub(upper: ndarray, b: ndarray) -> ndarray:
    """
    This is a backward substitution for upper triangular matrix. An efficient method to retrieve x by
    xA = b when A in upper triangular.
    :param upper: A Upper triangula matrix
    :param b: b vector
    :return: x vector
    """
    n, m = b.shape[0], upper.shape[0]
    x = np.zeros((n, m))

    for c in range(n):
        x[c, 0] = b[c, 0] / upper[0, 0]
        for i in range(1, m):

            acc = 0
            for j in range(i):
                acc += upper[j, i] * x[c, j]

            x[c, i] = (b[c, i] - acc) / upper[i, i]

    return x


def forwardSub(lower: ndarray, b: ndarray) -> ndarray:
    """
    This is a forward substitution for lower triangular matrix. An efficient method to retrieve x by
    xA = b when A in lower triangular.
    :param lower: A lower triangula matrix
    :param b: b vector
    :return: x vector
    """
    n, m = b.shape[0], lower.shape[0]
    x = np.zeros((n, m))

    for c in range(n):
        x[c, -1] = b[c, -1] / lower[-1, -1]

        for i in range(m - 1, -1, -1):
            acc = 0
            for j in range(i + 1, lower.shape[0]):
                acc += lower[j, i] * x[c, j]

            x[c, i] = (b[c, i] - acc) / lower[i, i]

    return x
