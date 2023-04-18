import numpy as np


def cross_operator(U):
    return np.array([[0, -U[2], U[1]],
                     [U[2], 0, -U[0]],
                     [-U[1], U[0], 0]])
