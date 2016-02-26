import numpy as np


def compute_gradient(J, theta):
    eps = 1e-6
    if not isinstance(theta, np.ndarray):
        l = []
        l.append(theta)
        theta = np.array(l) # in case J is 1 argument function
    args_count = theta.shape[0]
    gradient = np.empty(args_count)
    for i in range(args_count):
        add = np.zeros(args_count)
        add[i] = eps
        gradient[i] = (J(theta+add) - J(theta-add)) / (2*eps)
    return gradient


def check_gradient():
    def J1(x):
        return x**2
    theta = 20
    check1 = np.all(np.isclose(compute_gradient(J1, theta), 40))
    
    def J2(theta):
        return theta[0]**3 + theta[1]**2
    theta = np.array([20, 30])
    check2 = np.all(np.isclose(compute_gradient(J2, theta), [1200, 60]))
    
    return check1 and check2