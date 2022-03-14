import numpy as np
import copy


def make_matrix(A, x, b):
    B = copy.deepcopy(A)
    c = copy.deepcopy(b)

    diag_elem = np.diagonal(B)
    di_el = copy.deepcopy(diag_elem)

    for i, row in enumerate(B):
        for j in range(len(row)):
            B[i][j] = B[i][j] / di_el[i]
        c[i] = c[i] / di_el[i]

    B = B * -1
    for i, row in enumerate(B):
        B[i, i] = 0

    return B, c
