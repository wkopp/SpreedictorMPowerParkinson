import numpy as np

def rotX(angle):
    return np.array(
            [[1,        0,                  0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]])
def rotY(angle):
    return np.array(
            [[np.cos(angle), 0, np.sin(angle)],
            [0,              1,         0],
            [-np.sin(angle), 0, np.cos(alpha)]])
def rotZ(angle):
    return np.array(
            [[np.cos(angle), -np.sin(angle),    0],
            [np.sin(angle), np.cos(angle),      0],
            [0,                 0,              1]])

def rotate(alpha, beta, gamma):
    return np.matmul(np.matmul(rotX(alpha), rotY(beta)), rotZ(gamma))
