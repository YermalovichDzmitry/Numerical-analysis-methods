import numpy as np
import copy


def check_convergence_simple(B):
    kof1 = 0
    kof2 = 0
    kof3 = 0
    arr = []
    brr = []

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            kof1 = kof1 + B[i, j] ** 2
            kof2 = kof2 + abs(B[i, j])
            kof3 = kof3 + abs(B[j, i])
        arr.append(kof2)
        brr.append(kof3)
        kof2 = 0
        kof3 = 0

    kof1 = kof1 ** 0.5
    kof2 = max(arr)
    kof3 = max(brr)

    if kof1 < 1 or kof2 < 1 or kof3 < 1:
        print("Converge")
        return 1
    else:
        print("Not Converge")
        return 0


def check_convergence_seidel(B):
    kof2 = 0
    kof3 = 0
    arr = []
    brr = []

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            kof2 = kof2 + abs(B[i, j])
            kof3 = kof3 + abs(B[j, i])
        arr.append(kof2)
        brr.append(kof3)
        kof2 = 0
        kof3 = 0

    kof2 = max(arr)
    kof3 = max(brr)

    if kof2 < 1 or kof3 < 1:
        print("Converge")
        return 1
    else:
        print("Not Converge")
        return 0
