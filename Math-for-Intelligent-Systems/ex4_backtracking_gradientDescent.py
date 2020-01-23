"""
Backtracking line search: Gradient descent
"""

# import packages
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2D array C of size 2 * 2
C = np.array([[1, 0], [0, 10]])
def f_sq(x):
    """ Square function definition """
    return np.matmul(np.matmul(x.T, C), x)

def df_sq(x):
    """ Derivative of square function """
    return 2 * np.dot(C, x)

def f_hole(x):
    """ Hole function definition """
    return 1 - np.exp(-f_sq(x))

def df_hole(x):
    """ Derivative of hole function """
    return np.exp(-f_sq(x)) * df_sq(x)

def gradientdescent(x0, threshold,func, dfunc):
    """ Backtracking line search gradient descent: Assign constants, iterate until wolfe condition is met"""
    innerloop = 1
    outerloop = 1
    alpha = 1
    sigls = 0.01
    siginc = 1.2
    sigdec = 0.5
    neggrad = - dfunc(x0)
    step = neggrad / la.norm(neggrad)
    x = x0

    vals = []
    objectfs = []

    # print statements for debugging
    print (step)
    print(la.norm(step))
    print (func(x + alpha * step)>func(x) + sigls * np.dot(dfunc(x), alpha * step))
    print (func(x) + sigls * np.dot(dfunc(x), alpha * step))
    print (func(x + alpha * step))

    # condition of convergence or we can limit the number of iterations
    while la.norm(alpha*step) > threshold:
        neggrad = - dfunc(x)
        step = neggrad / la.norm(neggrad)

        while func(x + alpha * step) > func(x) + sigls * np.dot(dfunc(x), alpha * step):
            alpha = sigdec * alpha
            innerloop += 1
            print ("Dec alpha:", alpha)
        print ("Count: ", outerloop)
        x = x + step
        print ("x value:", x)
        alpha = siginc * alpha
        vals.append(x)
        objectfs.append(func(x))
        outerloop += 1
    print(x, func(x), outerloop)
    return vals, objectfs, outerloop

def plot_trace(xval, f):
    """ Plot function for a vector xval and function f """
    x = np.array([i[0] for i in xval])
    y = np.array([i[1] for i in xval])
    z = np.array(f)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, label='gradient descent method')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Initial starting point for descent
    x0 = np.array([1, 1])
    print ("Initial value:", x0)
    print ("Square function value:", f_sq(x0))

    # call gradient descent with initial value, threshold, function and its derivative
    x, f, iters = gradientdescent(x0, 0.1, f_sq, df_sq)

    # print and plot the results
    print (x)
    print (f)
    print (iters)
    plot_trace(x,f)

    # gradient descent for hole function
    val, objectf, iters = gradientdescent(x0, 0.1, f_hole, df_hole)

