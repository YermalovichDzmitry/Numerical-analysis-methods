import copy
import numpy as np


def check_diagonal(A):
    n = A.shape[0]
    stack = []
    for i in range(n):
        if A[i][i] == 0:
            stack.append(i)

    if len(stack) == 0:
        return 1

    for i in stack:
        for j in range(n):
            if A[j, i] != 0 and A[i, j] != 0:
                buf = copy.deepcopy(A[j, :])
                A[j, :] = A[i, :]
                A[i, :] = buf
                break
    stack.clear()

    for i in range(n):
        for j in range(n):
            if A[i][i] == 0:
                stack.append(i)

    if len(stack) != 0:
        return 0
    else:
        return 1
