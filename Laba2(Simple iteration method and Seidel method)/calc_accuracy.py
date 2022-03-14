import numpy as np


def calc_accuracy(x, x_prev):
    a = abs(max(x - x_prev, key=lambda item: abs(item)))
    v = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    trfl = [item > a for item in v]
    error = 0
    for i, item in enumerate(trfl):
        if not item:
            error = v[i - 1]
            break
    return error

# def calc_accuracy(x, x_prev, A, b):
#     return np.linalg.norm(b - np.dot(A, x))
