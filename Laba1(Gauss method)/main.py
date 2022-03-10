import numpy as np
import copy
from numpy import unravel_index


def check_compatibility(A, b):
    n, m = A.shape
    if n < m:
        print("The system is consistent and has infinitely many solutions")
        return 0
    elif n > m:
        print("The system is overridden i can't solve it")
        return 0

    main_det = np.linalg.det(A)
    dets = []
    for i in range(2):
        A_copy = copy.deepcopy(A)
        A_copy[:, i] = b_2
        dets.append(np.linalg.det(A_copy))
    if main_det == 0:
        for elem in dets:
            if elem != 0:
                print("Has no solutions")
                return 0
        print("Infinitely many solutions")
        return 0
    return 1


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


def print_answer(A, x, b):
    n = A.shape[0]
    for i in range(n):
        print(f"x{i} = {x[i]}")
    print(f"\nOriginal b = {b}")
    print(f"\nCheck array: b = {np.dot(A, x)}")


def first_way(A):
    # Приводим А к нижнетреугольной, прямой ход
    n = A.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n, 1):
            q = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - q * A[i, :]
            # Обратный ход
    x = np.array(np.zeros(n))
    kof = 0
    for i in range(n - 1, -1, -1):
        kof = np.dot(x, A[i, :n])
        x[i] = (A[i, n] - kof) / A[i, i]

    return x, n


def second_way(A):
    n = A.shape[0]

    # Прямой ход
    for i in range(n - 1):

        elem = np.max(A[i:n, i])
        index = np.where(A[i:n, i] == elem)

        buf = copy.deepcopy(A[index[0] + i, :])
        A[index[0] + i, :] = A[i, :]
        A[i, :] = buf

        for j in range(i + 1, n, 1):
            q = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - q * A[i, :]

    # Обратный ход
    x = np.array(np.zeros(n))
    kof = 0
    for i in range(n - 1, -1, -1):
        kof = np.dot(x, A[i, :n])
        x[i] = (A[i, n] - kof) / A[i, i]
    return x, n


def third_way(A):
    n = A.shape[0]
    stack = []
    for i in range(n - 1):

        coord = unravel_index(A[i:n, i:n].argmax(), A[i:n, i:n].shape)
        index, row = coord
        stack.append(row + i)

        buf = copy.deepcopy(A[index + i, :])
        A[index + i, :] = A[i, :]
        A[i, :] = buf

        for j in range(i + 1, n, 1):
            q = A[j, row + i] / A[i, row + i]
            A[j, :] = A[j, :] - q * A[i, :]

    sort_stack = np.sort(stack)
    ans = 0
    for i in range(n - 1):
        if sort_stack[i] != i:
            ans = i
            break
    stack.append(ans)

    x = np.array(np.zeros(n))
    for i in range(n - 1, -1, -1):
        index = stack.pop()
        s = 0
        for j in range(n):
            if j != index:
                s = s + x[j] * A[i, j]
        x[index] = (A[i, n] - s) / A[i, index]
    return x, n


def error_estimate(x, A, b, b_2):
    b_t = np.dot(A, x)
    x_copy = copy.deepcopy(x)
    delta_b = b_t - b_2

    A_expand = np.zeros((A.shape[0], A.shape[0] + 1))
    A_expand[:, :A.shape[0]] = A
    A_expand[:, A.shape[0]] = delta_b

    delta_x, n = first_way(A_expand)

    for i in range(len(delta_x)):
        delta_x[i] = abs(delta_x[i])
    absolute = max(delta_x)

    for i in range(len(x_copy)):
        x_copy[i] = abs(x_copy[i])

    relative = absolute / max(x_copy)
    return absolute, relative


def run(A, b, b_2, k):
    A_copy = copy.deepcopy(A)
    A_expand = copy.deepcopy(A)
    b_copy = copy.deepcopy(b)
    A_expand = np.hstack((A_expand, b))
    b_2 = copy.deepcopy(b_2)

    if not check_compatibility(A_copy, b_2):
        print("Матрица несовместна")
        return 0

    if not check_diagonal(A_expand):
        print("Can't solve")
        return 0

    if k == 1:
        x, n = first_way(A_expand)
    elif k == 2:
        x, n = second_way(A_expand)
    elif k == 3:
        x, n = third_way(A_expand)

    print_answer(A, x, b)

    A_copy = copy.deepcopy(A)
    A_expand = copy.deepcopy(A)
    b_copy = copy.deepcopy(b)
    b_2_copy = copy.deepcopy(b_2)

    absolute, rel = error_estimate(x, A_copy, b_copy, b_2_copy)
    # print(f"Absolute error ={absolute}\nRelative error={rel}")
    # print()
    kof = 1

    while rel < 0.0001:
        A_copy = copy.deepcopy(A)
        A_expand = copy.deepcopy(A)
        b_copy = copy.deepcopy(b)
        b_2_copy = copy.deepcopy(b_2)
        for i in range(len(x)):
            x[i] = round(x[i], 15 - kof)
        absolute, rel = error_estimate(x, A_copy, b_copy, b_2_copy)
        # print(f"Absolute error ={absolute}\nRelative error={rel}")
        # print()
        # print(x)
        # print()
        kof += 1
    # print("dasasda")
    print(x)

    return x


# b = np.array([[4.2], [4.2], [4.2], [4.2], [4.2]])
# b_2 = np.array([4.2, 4.2, 4.2, 4.2, 4.2])
# C = np.array([
#     [0.2, 0.0, 0.2, 0.0, 0.0],
#     [0.0, 0.2, 0.0, 0.2, 0.0],
#     [0.2, 0.0, 0.2, 0.0, 0.2],
#     [0.0, 0.2, 0.0, 0.2, 0.0],
#     [0.0, 0.0, 0.2, 0.0, 0.2]
# ])
# D = np.array([
#     [2.33, 0.81, 0.67, 0.92, -0.53],
#     [-0.53, 2.33, 0.81, 0.67, 0.92],
#     [0.92, -0.53, 2.33, 0.81, 0.67],
#     [0.67, 0.92, -0.53, 2.33, 0.81],
#     [0.81, 0.67, 0.92, -0.53, 2.33]
# ])
# A = 10 * C + D

# print(A)

# A = np.array([
#     [3, -4],
#     [3, -4]
# ])
# b = np.array([[12], [18]])
# b_2 = np.array([12, 18])  # не имеет решений

# A = np.array([
#     [7, 6],
#     [3.5, 3]
# ])
# b = np.array([[-42], [-21]])
# b_2 = np.array([-42, -21])  # бесконечно много решений

# A = np.array([
#     [7, 6, 6],
#     [3.5, 3, 3]
# ])
# b = np.array([[-42], [-21]])
# b_2 = np.array([-42, -21])  # система неопределена или имеет бесконечно много решений

# A = np.array([
#     [7, 6],
#     [3.5, 3],
#     [3.5, 3]
# ])
# b = np.array([[-42], [-21], [43]])
# b_2 = np.array([-42, -21, 43])  # система переопределена

# A = np.array([
#     [2.33, 0.81, 0.67],
#     [-0.53, 0, 1],
#     [0.92, -0.53, 0]
# ])
# b = np.array([[4], [21], [9]])
# b_2 = np.array([4, 21, 9])

b = np.array([[4.2], [4.1], [4.2], [4.2], [4.2]])
b_2 = np.array([4.2, 4.1, 4.2, 4.2, 4.2])

A = np.array([
    [4.33, 0.81, 2.67, 0.92, -0.53],
    [-0.53, 4.32, 0.81, 2.67, 0.92],
    [2.92, -0.53, 4.33, 0.81, 2.67],
    [0.67, 2.92, -0.53, 4.33, 0.81],
    [0.81, 0.67, 2.92, -0.54, 4.33]
])

x2 = run(A, b, b_2, 2)

b = np.array([[4.2], [4.2], [4.2], [4.2], [4.2]])
b_2 = np.array([4.2, 4.2, 4.2, 4.2, 4.2])
C = np.array([
    [0.2, 0.0, 0.2, 0.0, 0.0],
    [0.0, 0.2, 0.0, 0.2, 0.0],
    [0.2, 0.0, 0.2, 0.0, 0.2],
    [0.0, 0.2, 0.0, 0.2, 0.0],
    [0.0, 0.0, 0.2, 0.0, 0.2]
])
D = np.array([
    [2.33, 0.81, 0.67, 0.92, -0.53],
    [-0.53, 2.33, 0.81, 0.67, 0.92],
    [0.92, -0.53, 2.33, 0.81, 0.67],
    [0.67, 0.92, -0.53, 2.33, 0.81],
    [0.81, 0.67, 0.92, -0.53, 2.33]
])
A = 10 * C + D
x1 = run(A, b, b_2, 3)
print()
print(np.linalg.norm(x2-x1))