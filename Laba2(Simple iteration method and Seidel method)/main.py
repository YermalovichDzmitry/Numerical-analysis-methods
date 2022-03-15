import numpy as np
import copy
from check_convergence import check_convergence_simple
from check_convergence import check_convergence_seidel
from check_compatibility import check_compatibility
from check_diagonal import check_diagonal
from make_matrix import make_matrix
from simple_iteration_method import simple_iteration_method
from seidel_method import seidel_method
from print_answer import print_answer


def run(A, b, b_2, k):
    x = np.array([[1.2], [1.2], [1.2], [1.2], [1.2]])
    # x = np.array([[1.2], [1.2]])
    # x = np.array([[1.2], [1.2], [1.2]])
    n, m = A.shape
    A_copy = copy.deepcopy(A)
    A_expand = copy.deepcopy(A)
    b_copy = copy.deepcopy(b)
    A_expand = np.hstack((A_expand, b))

    if not check_compatibility(A_copy, b_2):
        print("СЛАУ не имеет решений или имеет бесконечно много")
        return 0

    if not check_diagonal(A_expand):
        print("Не могу решить систему")
        return 0

    b = A_expand[:, -1]
    A = A_expand[:, 0:n]
    b = np.expand_dims(b, axis=1)
    new_x = 0
    if k == 1:
        new_x, num_of_iter, error = simple_iteration_method(A, x, b)
    elif k == 2:
        new_x, num_of_iter, error = seidel_method(A, x, b)
    new_xa = []
    for i in range(len(new_x)):
        v = round(float(new_x[i]), 4)
        new_x[i] = v

    print_answer(A, new_x, b)
    print(f"{num_of_iter} iterations\nError={error}")
    return new_x


# b = np.array([[1.2], [2.2], [4.0], [0.0], [-1.2]])
# b_2 = np.array([1.2, 2.2, 4.0, 0.0, -1.2])
# C = np.array([
#     [0.01, 0, -0.02, 0, 0],
#     [0.01, 0.01, -0.02, 0, 0],
#     [0, 0.01, 0.01, 0, -0.02],
#     [0, 0, 0.01, 0.01, 0],
#     [0, 0, 0, 0.01, 0.01]
# ])
# D = np.array([
#     [1.33, 0.21, 0.17, 0.12, -0.13],
#     [-0.13, -1.33, 0.11, 0.17, 0.12],
#     [0.12, -0.13, -1.33, 0.11, 0.17],
#     [0.17, 0.12, -0.13, -1.33, 0.11],
#     [0.11, 0.67, 0.12, -0.13, -1.33]
# ])
# A = 7 * C + D

# A = np.array([
#     [2.33, 0.81, 0.67],
#     [-0.53, 0.0, 1.0],
#     [0.92, -0.53, 0.0]
# ])
# b = np.array([[4.0], [21.0], [9.0]])
# b_2 = np.array([4.0, 21.0, 9.0])


# A = np.array([
#     [0.92, -0.53, 1.0],
#     [2.33, 0.81, 0.67],
#     [-0.53, 3.0, 1.0]
# ])
# b = np.array([[9.0], [4.0], [21.0]])
# b_2 = np.array([9.0, 4.0, 21.0])

# x = np.array([[1.2], [1.2], [1.2]])
# A = np.array([
#     [2.0, 1.0],
#     [1.0, -2.0]
# ])
# b = np.array([[3.0], [1.0]])
# b_2 = np.array([3.0, 1.0])
# x = np.array([[1.2], [1.1]])

# A = np.array([
#     [5.0, 2.0],
#     [2.0, 1.0]
# ])
# b = np.array([[7.0], [9.0]])
# b_2 = np.array([7.0, 9.0])
# x = np.array([[1.2], [1.1]])

# A = np.array([
#     [2.0, 1.0, 1.0],
#     [1.0, -1.0, 0.0],
#     [3.0, -1.0, 2.0]
# ])
# b = np.array([[2.0], [-2.0], [2.0]])
# b_2 = np.array([2.0, -2.0, 2.0])
# x = np.array([[1.2], [1.1]])

################################################################################
# A = np.array([
#     [2.0, 2.0, 3.0],
#     [-1.0, 2.0, 1.0],
#     [2.0, -4.0, -2.0]
# ])
# b = np.array([[1.0], [2.0], [1.0]])
# b_2 = np.array([1.0, 2.0, 1.0])

# A = np.array([
#     [4.0, -8.0],
#     [1.0, -2.0]
# ])
# b = np.array([[0.0], [0.0]])
# b_2 = np.array([0.0, 0.0])

# A = np.array([
#     [-1, 0.5, 0.6],
#     [0, -1, 0.5],
#     [0.5, 0, -1]
# ])
# b = np.array([[1], [1], [1]])
# b_2 = np.array([1, 1, 1])
#
# x1 = run(A, b, b_2, 2)
#
# print("====================")
#
# A = np.array([
#     [-1, 0.51, 0.6],
#     [0, -1.01, 0.5],
#     [0.51, 0, -1]
# ])
# b = np.array([[1], [1], [1]])
# b_2 = np.array([1, 1, 1])
# x2 = run(A, b, b_2, 2)
# b = np.array([[1.2], [2.2], [4.0], [0.0], [-1.2]])
# b_2 = np.array([1.2, 2.2, 4.0, 0.0, -1.2])
# C = np.array([
#     [0.01, 0, -0.02, 0, 0],
#     [0.01, 0.01, -0.02, 0, 0],
#     [0, 0.01, 0.01, 0, -0.02],
#     [0, 0, 0.01, 0.01, 0],
#     [0, 0, 0, 0.01, 0.01]
# ])
# D = np.array([
#     [1.33, 0.21, 0.17, 0.12, -0.13],
#     [-0.13, -1.33, 0.11, 0.17, 0.12],
#     [0.12, -0.13, -1.33, 0.11, 0.17],
#     [0.17, 0.12, -0.13, -1.33, 0.11],
#     [0.11, 0.67, 0.12, -0.13, -1.33]
# ])
# A = 8 * C + D
# print(A)
b = np.array([[1.2], [2.2], [4.0], [0.0], [-1.2]])
b_2 = np.array([1.2, 2.2, 4.0, 0.0, -1.2])
A = np.array([
    [1.41, 0.21, 0.01, 0.12, -0.13],
    [-0.05, -1.25, -0.05, 0.17, 0.12],
    [0.12, -0.05, -1.25, 0.11, 0.01],
    [0.17, 0.12, -0.05, -1.25, 0.11],
    [0.11, 0.67, 0.12, -0.05, -1.25]
])
x1 = run(A, b, b_2, 1)
print("===================")

b = np.array([[1.2], [2.2], [4.0], [0.0], [-1.2]])
b_2 = np.array([1.2, 2.2, 4.0, 0.0, -1.2])
A = np.array([
    [1.41, 0.21, 0.01, 0.12, -0.13],
    [-0.05, -1.25, -0.05, 0.18, 0.12],
    [0.12, -0.06, -1.25, 0.11, 0.01],
    [0.17, 0.12, -0.05, -1.25, 0.11],
    [0.12, 0.67, 0.12, -0.05, -1.25]
])
x2 = run(A, b, b_2, 1)
print(np.linalg.norm(x2 - x1))
