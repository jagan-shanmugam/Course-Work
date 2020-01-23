""" finite difference gradient checker
 MIS exercise """

# required packages
import numpy as np
from numpy import linalg as la

# constants
n=3
epsilon = 1e-6

# generate random vector and matrix
np.random.seed(1)
x=np.random.randn(n,1)
A=np.random.randn(n,n)

# generate basis for required dimension(n)
e = np.eye(n)

""" # print values of basis, x and A to verify. 
print("basis e",e)
print ("x: ",x)
print ("A: ",A)
"""

# empty Jacobian matrices
J1 = np.empty((n,n))
J2 = np.empty((n,n))

def func1(x):
    """ function 1: A * x """
    return np.dot(A,x)

def func2(x):
    """ function 2: x.T * x """
    return np.dot(np.array(x)[:,0], x)

def df1(x):
    """ derivative of function 1 """
    return A

def df2(x):
    """ derivative of function 2 """
    return 2*(x.T)

def numericalgrad(ex,J):
    """ numerical gradient checker  function :
    iterate through dimension n and compute finite differences of the function """
    for i in range(n):
        xplus=(x.T)+(e[i]*epsilon)
        print ("xplus ",xplus)
        xminus=(x.T)-(e[i]*epsilon)
        print ("xminus", xminus)
        if ex==1:
            J[:,i]=((func1(xplus.T)-func1(xminus.T))/(2*epsilon)).T
        else:
            J[:,i]=((func2(xplus.T) - func2(xminus.T)) /(2*epsilon)).T

# call the numerical gradient function for problem 1 and 2
numericalgrad(1,J1)
numericalgrad(2,J2)

# print the values of jacobians
print ("J1 ",J1)
print ("df1: ",df1(x))
print ("J2 ",J2)
print ("df2: ",df2(x))

# compare and print results for problem 1 and problem 2
print ("Problem 1")
if(la.norm(J1-df1(x),np.inf)<1e-4):
    print ("True")
else:
    print ("False")

print ("Problem 2")
if(la.norm(J2-df2(x),np.inf)<1e-4):
    print ("True")
else:
    print ("False")


