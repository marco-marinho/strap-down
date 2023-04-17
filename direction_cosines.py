import numpy as np
from numpy.linalg import inv, norm
from operators import cross_operator


def get_direction_cosine(U_1, V_1, U_2, V_2):
    U_1 = np.array(U_1)
    V_1 = np.array(V_1)
    W_1 = np.cross(U_1, V_1)
    U_2 = np.array(U_2)
    V_2 = np.array(V_2)
    W_2 = np.cross(U_2, V_2)
    D_1 = np.c_[U_1, V_1, W_1]
    D_2 = np.c_[U_2, V_2, W_2]
    return D_1 @ inv(D_2)


def get_direction_cosine_single(U_1, U_2):
    U_1 = np.array(U_1)
    U_2 = np.array(U_2)
    V = norm(U_1)
    D = (1 * (V**2))*(U_2.T @ U_1)
    E = (1 * (V**2))*np.cross(U_2, U_1)
    Ex = cross_operator(E)
    return np.identity(3) + Ex + (1/(1 + D)) * (Ex @ Ex)

