# Surface plot

# Required packages
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D

# Function for quadratic equation
def fsq_2D(c1,c2,x, y, temp=100):
  #return np.sqrt(2 - c1*(x**2) - c2*(y**2))
  return np.exp(x/temp) / (np.exp(x/temp) + np.exp(y/temp))

# vary constants c1 and c2
c1=0.1
c2=0.1

# create a figure and add a plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# prepare x and y axis data
x = y = np.arange(-500.0, 500.0, 0.10)
X, Y = np.meshgrid(x, y)

# for values of x and y, compute the value of the quadratic function
zs = np.array([fsq_2D(c1,c2,x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

# plot the surface
ax.plot_surface(X, Y, Z)

# set labels and display
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('fsq')
plt.show()