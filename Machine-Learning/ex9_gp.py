
import numpy as np
from matplotlib import pyplot as plt
l = 0.2
gamma = 1
sigma = 0.1
lam = sigma ** 2

def cov(x, x1):
    """ obsolete """
    d = (x - x1) / l
    print("d",d)
    d = np.linalg.norm(d,gamma).reshape(-1,1)
    return np.exp(-1 * d)

def c(x1,x2):
    return np.exp(-1 * (np.subtract.outer(x1, x2) / l) ** gamma)

n = 100
N = 2
#X = np.asarray([-0.5, 0.5]).reshape(N,1)
#print(X.shape)
x = [-0.5, 0.5]
y = np.asarray([0.3, -0.1])
print(y.shape)

""" test
K = cov(X[0], X[0])
print(K)
K = cov(X[0], X[1])
print(K)
K = cov(X[1], X[0])
print(K)
K = cov(X[1], X[1])
print(K)
"""

#print(c(x,x))
Xtest = np.linspace(-5, 5, n)
K = c(x, x)
print(K.shape)

inv = np.linalg.inv(K + lam * np.eye(len(K)))

#kappa = c(x,Xtest)
#print(inv.shape)
kappa = c(Xtest,x)
print(kappa.shape)

mu = np.matmul(kappa,np.matmul(inv, y))
print(mu.shape)
print(c(Xtest,Xtest).shape)

var = (sigma **2 / lam) * c(Xtest,Xtest) - (sigma**2/lam) * np.matmul(kappa, np.matmul(inv,kappa.T))
sd = np.sqrt(np.diag(var))
print(sd.shape)

plt.plot(x, y, 'ro')
plt.fill_between(Xtest.flat, mu-sd, mu+sd, color="#dddddd")
#plt.errorbar(Xtest, y, yerr=var, capsize=0)
plt.show()