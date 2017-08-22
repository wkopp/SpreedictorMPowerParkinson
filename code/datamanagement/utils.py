import numpy as np

def __rotX(angle):
    return np.array(
            [[1,        0,                  0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]])
def __rotY(angle):
    return np.array(
            [[np.cos(angle), 0, np.sin(angle)],
            [0,              1,         0],
            [-np.sin(angle), 0, np.cos(angle)]])
def __rotZ(angle):
    return np.array(
            [[np.cos(angle), -np.sin(angle),    0],
            [np.sin(angle), np.cos(angle),      0],
            [0,                 0,              1]])

def __rotate(angles):
    return np.matmul(np.matmul(__rotX(angles[0]), __rotY(angles[1])), __rotZ(angles[2]))

def randomRotation(timeseries):
    """Rotate the timeseries.

    A timeseries matrix ``steps x coord`` will be
    randomly rotated by +/- 10 degree around each coordinate axis.
    """

    angles = np.random.uniform(-np.pi/2.*(1./9), np.pi/2.*(1./9), size = 3)

    return np.matmul(timeseries, __rotate(angles).T)

def batchRandomRotation(timeseries):
    """Apply :func:`randomRotation` to all samples."""

    for t in range(timeseries.shape[0]):
        timeseries[t] = randomRotation(timeseries[t])

    return timeseries
