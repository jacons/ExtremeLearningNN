import numpy as np
from numpy import ndarray, zeros_like, sqrt


def cholesky(matrix: ndarray) -> ndarray:
    """
    This function implements the Cholesky decomposition in numpy.
    The matrix is positive definitive can be decomposed in A = "R.T x R" where
    R is "upper triangular". The complexity is equal to O((n^3))
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


def backwardSub(A: ndarray, b: ndarray) -> ndarray:
    """
    This is a backward substitution for upper triangular matrix. An efficient method to retrieve x by
    xA = b when A in upper triangular.
    :param A: A Upper triangula matrix
    :param b: b vector
    :return: x vector
    """
    n, m = b.shape[0], A.shape[0]
    x = np.zeros((n, m))

    for c in range(n):
        x[c, 0] = b[c, 0] / A[0, 0]
        for i in range(1, m):

            acc = 0
            for j in range(i):
                acc += A[j, i] * x[c, j]
            x[c, i] = (b[c, i] - acc) / A[i, i]

    return x


def forwardSub(A: ndarray, b: ndarray) -> ndarray:
    """
    This is a forward substitution for lower triangular matrix. An efficient method to retrieve x by
    xA = b when A in lower triangular.
    :param A: A lower triangula matrix
    :param b: b vector
    :return: x vector
    """
    n, m = b.shape[0], A.shape[0]
    x = np.zeros((n, m))

    for c in range(n):
        x[c, -1] = b[c, -1] / A[-1, -1]

        for i in range(m - 1, -1, -1):
            acc = 0
            for j in range(i + 1, A.shape[0]):
                acc += A[j, i] * x[c, j]
            x[c, i] = (b[c, i] - acc) / A[i, i]

    return x
