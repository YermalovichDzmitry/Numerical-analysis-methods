from make_matrix import make_matrix
from check_convergence import check_convergence_simple
import numpy as np
from calc_accuracy import calc_accuracy


def simple_iteration_method(A, x, b):
    B, c = make_matrix(A, x, b)
    check_convergence_simple(B)

    error = 100
    num_of_iter = 0
    x_prev = x
    while error > 0.0001:
        x = np.dot(B, x) + c
        error = calc_accuracy(x, x_prev)
        x_prev = x
        num_of_iter += 1
    return x, num_of_iter, error
