import numpy as np
from numpy import linalg as la

epsilon = 1e-6
(h0,h1,h2,h3)=(5,10,10,2)
n=h0
np.random.seed(1)
e = np.eye(n)

x0=np.random.rand(h0,1)
print ("x: ",x0)

# initialize weights randomly
w0 = np.random.rand(h1, h0)
w1 = np.random.rand(h2, h1)
w2 = np.random.rand(h3, h2)

J = np.empty((h3,h0))
#print ("Initial J ",J)

# sigmoid function
def sigmoid(z, deriv=False):
    if (deriv == True):
        return np.diagflat(np.multiply(z, 1 - z))
    return 1 / (1 + np.exp(-z))

def f(x0):
    z1 = np.dot(w0,x0)
    #print ("z1: ", z1)
    x1 = sigmoid(z1)
    #print ("x1: ", x1)
    z2 = np.dot(w1, x1)
    #print ("z2: ", z2)
    x2 = sigmoid(z2)
    #print ("x2: ", x2)
    f = np.dot(w2,x2)
    return f


def df(x0):
    dz1=w0
    z1=np.dot(w0,x0)
    x1=sigmoid(z1)
    dx1=np.dot(sigmoid(x1,True),dz1)
    dz2=np.dot(w1,dx1)
    z2=np.dot(w1,x1)
    x2=sigmoid(z2)
    dx2=np.dot((sigmoid(x2,True)),dz2)
    df_x=np.dot(w2,dx2)
    return df_x

def numericaljacobian(x):
    for i in range(n):
        #xplus=x
        #xminus=x
        xplus=(x.T)+(e[i]*epsilon)
        #print ("xplus ",xplus)
        xminus=(x.T)-(e[i]*epsilon)
        #print ("xminus", xminus)
        J[:,i]=((f(xplus.T)-f(xminus.T))/(2*epsilon)).T
    return J


print ("f: ",f(x0))
print ("df: ",df(x0))
J=numericaljacobian(x0)
print ("J=",J)

if (la.norm(J - df(x0), np.inf) < 1e-4):
    print ("True")
else:
    print ("False")

"""
df=', array([[ 0.09307289,  0.10143094,  0.07033218,  0.07677957,  0.08623619],
       [ 0.07060583,  0.08479089,  0.06080483,  0.05964115,  0.06813189]]))
"""

