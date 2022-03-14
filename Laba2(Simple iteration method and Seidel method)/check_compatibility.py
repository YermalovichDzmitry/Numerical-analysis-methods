import numpy as np
import copy


def check_compatibility(A, b_2):
    n, m = A.shape

    main_det = np.linalg.det(A)
    dets = []
    for i in range(n):
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
