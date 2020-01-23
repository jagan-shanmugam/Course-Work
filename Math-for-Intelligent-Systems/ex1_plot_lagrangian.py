""" MIS exercise
 plot lagrangian for the given function
"""
# packages
import numpy as np
import matplotlib.pyplot as plt

# function to optimize
def f(x):
    return x**2 + 1

# constraint function
def c(x):
    return (x-2)*(x-4)

x=np.linspace(-10, 10, 101)

# lagrangian function
def lagrangian(x,lam):
    return f(x) + lam * (c(x))

# dual function
def dual(lam):
    return 4 - (2 *lam)

fig = plt.figure()
ax = []
colors = ('k', 'r', 'b')

# function plot
plt.plot(x, f(x), label='f(x)')
plt.legend(loc='upper left')

# constraint function plot
plt.plot(x, c(x), label='c(x)')
plt.legend(loc='upper left')

plt.plot(x, lagrangian(x,0.5), label='l(x,0.5)')
plt.legend(loc='upper left')
plt.show()

# plot lagrangian of the function along with the function
i = 0
for i in range(20):
    ax.append(plt.axes())
    print (colors[i % 3]+'o')
    lam = i/2
    plt.plot(x,lagrangian(x, lam), colors[i % 3], label='lagrangian for lamda')
    ax[i].set(autoscale_on=True)
plt.show()
ax = []

# plot the dual function
i=0
for i in range(50):
    ax.append(plt.axes())
    print (colors[i % 3]+'o')
    plt.plot(i, dual(i), colors[i % 3]+'o', label='just lagrangian')
    ax[i].set(autoscale_on=True)
plt.show()
