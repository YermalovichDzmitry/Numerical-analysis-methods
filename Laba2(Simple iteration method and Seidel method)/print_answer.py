import numpy as np


def print_answer(A, x, b):
    n = A.shape[0]
    for i in range(n):
        print(f"x{i} = {x[i]}")
    print(f"\nOriginal b = {b}")
    print(f"\nCheck array: b = {np.dot(A, x)}")
