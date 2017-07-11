import numpy as np


def quat_mult(q, v):

    x,y,z,w = q

    s = np.linalg.norm(q)**(-2)

    M = np.array(
        [[1 - 2*s*(y*y + z*z),     2*s*(x*y - w*z),     2*s*(x*z + w*y)],
         [    2*s*(x*y + w*z), 1 - 2*s*(x*x + z*z),     2*s*(y*z - w*x)],
         [    2*s*(x*z - w*y),     2*s*(y*z + w*x), 1 - 2*s*(x*x + y*y)]])

    return np.dot(M,v)


def axisangle_to_q(v, theta):
    v = v / np.linalg.norm(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return (x, y, z, w)

if __name__ == '__main__':

    q = axisangle_to_q([1,1,1], np.pi*2/3)

    v = np.array([1,0,0])
    v= v.reshape((3,1))

    v = quat_mult(q,v)
    print v

    v = quat_mult(q,v)
    print v

    v = quat_mult(q,v)
    print v