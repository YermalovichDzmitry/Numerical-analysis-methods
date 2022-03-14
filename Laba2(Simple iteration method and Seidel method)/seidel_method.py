from make_matrix import make_matrix
from check_convergence import check_convergence_seidel
import numpy as np
from calc_accuracy import calc_accuracy


def seidel_method(A, x, b):
    B, c = make_matrix(A, x, b)

    check_convergence_seidel(B)

    E = np.eye(B.shape[0])
    H = np.tril(B)
    F = np.triu(B)

    error = 100
    num_of_iter = 0
    mat = np.linalg.inv(E - H)
    x_prev = x
    while error > 0.0001:
        x = np.dot(np.dot(mat, F), x) + np.dot(mat, c)
        error = calc_accuracy(x, x_prev)
        x_prev = x
        num_of_iter += 1
    return x, num_of_iter, error
