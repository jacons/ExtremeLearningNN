import numpy as np
from numpy import ndarray, zeros_like, sqrt


def cholesky(matrix: ndarray) -> ndarray:
    """
    This function implements the Cholesky decomposition in numpy. The Cholesky decomposition is a factorization of a
    positive definite matrix into the product of a lower triangular matrix and its transpose. For a matrix A, it
    can be decomposed as A = "R.T x R", where R is an upper triangular matrix. The complexity of this algorithm
    is O((n^3)), where n is the dimension of the matrix.

    Parameters:
     * matrix (ndarray): Positive definite square matrix.

    Returns:
     * ndarray: Upper triangular matrix R from the Cholesky decomposition A = "R.T x R".
    """
    r = zeros_like(matrix)  # Upper triangula matrix
    n = matrix.shape[0]  # matrix dimension

    for j in range(n):
        for i in range(j, n):
            if i == j:
                # Diagonal elements of R
                r[i, j] = sqrt(matrix[i, j] - np.sum(r[i, :] ** 2))
            else:
                # Off-diagonal elements of R
                r[i, j] = (matrix[i, j] - np.sum(r[i, :] * r[j, :])) / r[j, j]

    return r  # Upper triangula matrix


def backwardSub(A: ndarray, b: ndarray) -> ndarray:
    """
    This function performs backward substitution for an upper triangular matrix. Backward substitution is an efficient
    method to solve a system of linear equations when the matrix A is upper triangular. It starts with the last
    equation and solves for the corresponding variable, then proceeds backward to find the other variables.

    Parameters:
     * A (ndarray): Upper triangular matrix.
     * b (ndarray): b vector.

    Returns:
     * ndarray: x vector, solution to the system xA = b.
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
    This function performs forward substitution for a lower triangular matrix. Forward substitution is an efficient
    method to solve a system of linear equations when the matrix A is lower triangular. It starts with the first
    equation and solves for the corresponding variable, then proceeds forward to find the other variables.

    Parameters:
     * A (ndarray): Lower triangular matrix.
     * b (ndarray): b vector.

    Returns:
     * ndarray: x vector, solution to the system xA = b.
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
