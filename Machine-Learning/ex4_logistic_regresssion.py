import numpy as np
from numpy.linalg import inv
from functools import reduce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
dim = 2
lam = 10             #Vary lamda
featureflag = 1     #LINEAR FEATURES set - 1 ; QUADRATIC FEATURES - 2 parameter
probflag = 1        #Set to 1 to view probability function  ; else discriminative function

def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate.
    np.meshgrid is pretty annoying!"""
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])

def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

def diag(z):
    return np.diagflat(np.multiply(z, 1-z))

def sigmoid(z):
    return np.array(1 / (1 + np.exp(-z)))

def prepend_one(X):
    """prepend a one vector to X."""
    return np.column_stack([np.ones(X.shape[0]), X])

def features(linflag,X):
    if linflag==1:
        X = prepend_one(X)
        return X
    elif linflag==2:
        X = np.column_stack([X, (X[:, 0] * X[:, 0]), (X[:, 0] * X[:, 1]), (X[:, 1] * X[:, 1])])
        X = prepend_one(X)
        return X

def gradient(X,y,beta,I):
    p = sigmoid(np.matmul(X,beta))
    return np.matmul(X.T, (p-y)) + 2 * lam * np.matmul(I,beta)

def hessian(X,I):
    p = sigmoid(np.matmul(X, beta))
    return mdot(X.T,diag(p),X) + 2*lam * I

def newton_descent(beta,y,I):
    beta[:] = 0
    print ("beta init:",beta)
    for i in range(100):
    #    print ("Iteration:",i," Beta",beta)
        grad = gradient(X, y, beta, I)
        hess = hessian(X, I)
        beta = beta - np.matmul(inv(hess), grad)
    return beta

def plot_prob(X,y,opti,linflag,probflag):
    X_grid = prepend_one(grid2d(-2, 3, num=30))
    if linflag == 2:
        X_grid = np.column_stack([X_grid, X_grid[:, 1] * X_grid[:, 1], X_grid[:, 1] * X_grid[:, 2], X_grid[:, 2] * X_grid[:, 2]])

    if probflag == 1:
        y_grid = sigmoid(np.matmul(X_grid, opti))
    else:
        y_grid = mdot(X_grid, opti)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # the projection part is important
    ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don’t use the 1 infront
    ax.scatter(X[:, 1], X[:, 2], y, color="red")  # also show the real data
    ax.set_title("Proability y=1 over test grid")
    plt.show()

def mean_neg_log_likelihood(X,beta,n):
    return (-1 * np.log(sigmoid(np.matmul(X,beta))) /n)

#read data
data = np.loadtxt("data2Class.txt")
X, y = data[:, :dim], data[:,dim]

#Number of data points
n=X.shape[0]

#LINEAR FEATURES paramter-1 ; QUADRATIC FEATURES parameter-2
X = features(featureflag,X)

I = np.eye(X.shape[1])
#Identity set element [0,0] to 0
I[0,0]=0
beta = np.empty(X.shape[1])

grad = gradient(X,y,beta,I)
print("Grad:",grad)

hess = hessian(X,I)
print("Hessian:",hess)

opti = newton_descent(beta,y,I)
print("Optimum beta:",opti)
print("lambda:",lam)

mean_nll = mean_neg_log_likelihood(X,opti,n)
print("Mean neg log likelihood:",mean_nll.mean())

plot_prob(X,y,opti,featureflag,probflag)

print("X.shape:", X.shape)
print("y.shape:", y.shape)


