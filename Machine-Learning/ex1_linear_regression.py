
"""
This is a mini demo of how to use numpy arrays and plot data.
NOTE: the operators + - * / are element wise operation. If you want
matrix multiplication use ‘‘dot‘‘ or ‘‘mdot‘‘!
"""


from numpy import linalg as la
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from functools import reduce
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.axes3d import Axes3D


###############################################################################

# Helper functions
def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

def prepend_one(X):
    """prepend a one vector to X."""
    return np.column_stack([np.ones(X.shape[0]), X])

def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate.
    np.meshgrid is pretty annoying!"""
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])

def quad_features(X):
    te = np.column_stack([X, (X[:, 0] * X[:,0]), (X[:, 0] * X[:,1])])
    return np.column_stack([ te, ( X[:,1] * X[:,1] ) ])
    # (X[:, 1] * X[:, 1])

###############################################################################
lam = 0.6
i = 2

if i==1:
    # part a
    data = np.loadtxt("dataLinReg2D.txt")
    #lam = np.linspace(0.001, 1000, 1000)
    print ("data.shape:", data.shape)
    np.savetxt("tmp.txt", data) # save data if you want to
    # split into features and labels
    X, y = data[:, :2], data[:, 2]
    X = prepend_one(X)
    I = np.eye(X.shape[1])
    #Identity set element [0,0] to 0
    I[0,0]=0
    print (I)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)

    print("lambda:",lam)

    # Fit model/compute optimal parameters beta
    beta_ = mdot(inv(dot(X.T, X) + lam * I), X.T, y)
    print("Optimal beta:", beta_)

    # prep for prediction
    X_grid = prepend_one(grid2d(-3, 3, num=30))
    print("X_grid.shape:", X_grid.shape)

    # Predict with trained model
    y_grid = mdot(X_grid, beta_)
    print("Y_grid.shape", y_grid.shape)

    err = la.norm(y - mdot(X, beta_))
    print ("Squared error for lin regression with Regularization:",err*err)

    # vis the result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection part is important
    ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don’t use the 1 infront
    ax.scatter(X[:, 1], X[:, 2], y, color="red")  # also show the real data
    ax.set_title("Linear regression predicted plane after regularization")
    plt.show()

elif i==2 :
    # part b
    #lam = np.linspace(0.001, 1000, 1000)
    data = np.loadtxt("dataQuadReg2D.txt")
    X_quad, y_quad= data[:, :2], data[:, 2]
    X_quad = quad_features(X_quad)
    X_quad = prepend_one(X_quad)
    X = X_quad
    y = y_quad
    I = np.eye(X.shape[1]) #Identity set element [0,0] to 0
    I[0, 0] = 0
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)

    print("lambda:",lam)

    # Fit model/compute optimal parameters beta
    beta_ = mdot(inv(dot(X.T, X) + lam * I), X.T, y)
    print ("Optimal beta:", beta_)

    err = la.norm(y - mdot(X, beta_))
    print("Squared error for Quad regression with Regularization:", err * err)

    # prep for prediction
    X_grid = prepend_one(grid2d(-3, 3, num=30))
    X_grid = np.column_stack([X_grid, X_grid[:, 1] * X_grid[:, 1], X_grid[:, 1] * X_grid[:, 2], X_grid[:, 2] * X_grid[:, 2]])
    print("X_grid.shape:", X_grid.shape)

    # Predict with trained model
    y_grid = mdot(X_grid, beta_)
    print("Y_grid.shape", y_grid.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection part is important
    ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don’t use the 1 infront
    ax.scatter(X[:, 1], X[:, 2], y, color="red")  # also show the real data
    ax.set_title("Quad regression predicted surface, after regularization")
    plt.show()

else:
    # part c : Cross validation
    lam = np.linspace(0.001, 1000, 1000)
    #lam = 0.9
    mse = np.empty(lam.size)
    mse_training = np.empty(lam.size)
    data = np.loadtxt("dataQuadReg2D_noisy.txt")
    for l in range(lam.size):
        k=5
        fold = 10
        m=0
        err = np.empty(5)
        err_training = np.empty(5)
        for i in range(k):
            x_test = data[m:m + fold, :2]
            x = np.concatenate((data[m - (fold * i):m, :2], data[m + fold:, :2]), axis=0)
            #5 folds of each 10
            y_test = data[m:m + fold, 2]
            y = np.concatenate((data[m - (fold * i):m, 2], data[m + fold:, 2]), axis=0)

            x = quad_features(x)
            x = prepend_one(x)
            x_test = quad_features(x_test)
            x_test = prepend_one(x_test)
            I = np.eye(x.shape[1]) #Identity set element [0,0] to 0
            I[0, 0] = 0

            print ("Training data")
            print (x.shape)
            print (y.shape)

            print ("Testing data")
            print(x_test.shape)
            print(y_test.shape)

            # compute beta_
            # Fit model/compute optimal parameters beta
            beta_ = mdot(inv(dot(x.T, x) + lam[l] * I), x.T, y)
            print("Optimal beta:", beta_)

            print("Error for cross validation, quad regression, training error")
            print(mdot((y - mdot(x, beta_).T), (y - mdot(x, beta_))))
            print(la.norm(y - mdot(x, beta_)))
            err_training[i] = la.norm(y - mdot(x, beta_))
            err_training[i] = err_training[i] * err_training[i]
            err_training[i] = err_training[i] / 40
            #compute test error
            print ("Test error")
            err[i] = la.norm(y_test - mdot(x_test, beta_))
            err[i] = err[i] * err[i]
            err[i] = err[i]/10
            print (err[i])
            m = m + 10

        #mean squared error
        print ("Mean squared error:")
        print (err.mean())
        mse[l] = err.mean()
        mse_training[l] = err_training.mean()

    #plot bar graph, lam and mse
    index = np.argmin(mse)
    print ("Min lamda:",lam[index])
    fig = plt.figure()
    print("Training Minimum mean squared error", mse.min())
    #mse = mse / mse.max()
    plt.plot(lam, mse,color='blue')
    plt.ylabel('Mean squared error')
    plt.xlabel('Lamda')
    plt.title('MSE for Lam')
    #plt.show()

    plt.plot(lam,mse_training,color='green')
    plt.xscale('log')
    #plt.ylabel('Training Mean squared error')
    #plt.xlabel('Lamda')
    plt.show()


# 3D plotting
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d') # the projection arg is important!
#ax.scatter(X[:, 0], X[:, 1], y, color="red")
#ax.set_title("raw data")
#plt.draw()
