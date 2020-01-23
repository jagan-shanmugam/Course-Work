#backprop

from matplotlib import pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D


def sigmoid(z, deriv=False):
    if (deriv == True):
        return sigmoid(z) * (1 - sigmoid(z))
        #return np.diagflat(np.multiply(z, 1 - z))
    #return expit(z)
    return 1 / (1 + np.exp(-z))


def plotdata(data):
    #print(data)
    one = data[data[:,3]==1]
    minus = data[data[:,3]==-1]
    plt.plot(one[:,1],one[:,2],'bo')
    plt.plot(minus[:,1],minus[:,2],'r*')
    plt.show()
    #2d plot
    plt.plot(data[:, 1], data[:, 3], 'ro')
    plt.plot(data[:, 2], data[:, 3], 'bo')
    plt.show()

def forward(x, w0, w1):
    #print(x.shape) # 200x3
    z1 = np.matmul(x, w0)
    x1 = sigmoid(z1)
    #print(x1.shape) #200x100
    out = np.matmul(x1, w1)
    #print(out.shape) #200x1
    return out

def gradientdescent(w0,w1,gradient_dw0, gradient_dw1):
    step = 0.05
    w0 = w0 - step * gradient_dw0
    w1 = w1 - step * gradient_dw1
    return w0, w1

def backward(y, x, w0, w1,estimate):
    hingeloss = np.zeros(len(y))
    delta_l2 = np.zeros([len(y),1])

    ones = np.ones([len(y),1])
    hinge = ones - np.multiply(y,estimate)
    hinge[hinge < 0] = 0

    for i in range(len(y)):
        tmp = 1 - y[i] * estimate[i]
        hingeloss[i] = max(0, tmp)
        if tmp > 0:
            delta_l2[i] = -y[i]
    hingeloss = hinge

    z1 = np.matmul(x, w0)
    x1 = sigmoid(z1)
    #print("x1:",x1.shape) #200x2
    diff_w1 = np.matmul(x1.T, delta_l2)
    #print("dl/dw1:",diff_w1.shape)

    #delta_l1 = np.multiply(np.matmul(w1.T, delta_l2.T), sigmoid_derv(z1).T)
    delta_l1 = np.matmul(delta_l2, w1.T) * sigmoid(z1, deriv=True)
    #print("del 1:", delta_l1.shape)
    diff_w0 = np.matmul(x.T, delta_l1)
    #print("dl/dw0:",diff_w0.shape)
    return diff_w0, diff_w1, hingeloss

def plot3d(data, estimate):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)
    # zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    # Z = zs.reshape(X.shape)
    Z = sigmoid(estimate)
    ax.scatter(data[:, 1], data[:, 2], Z)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f')
    plt.show()

def main():
    data = np.loadtxt('data2Class_adjusted.txt')
    print("data shape", data.shape)

    input_dim = 3
    hiddenlayersize = 100
    weights_hidden = np.asarray([[random.uniform(-1, 1) for i in range(input_dim)] for j in range(hiddenlayersize)])
    weights_hidden = weights_hidden.reshape(input_dim,hiddenlayersize)
    print("W0:", weights_hidden.shape)
    weights_output = np.asarray([random.uniform(-1, 1) for i in range(hiddenlayersize)])
    weights_output = weights_output.reshape(hiddenlayersize,1)
    print("W1:", weights_output.shape)

    #weights_hidden = np.zeros(weights_hidden.shape)
    #weights_output = np.zeros(weights_output.shape)

    x = data[:, :3]
    y = data[:, 3].reshape(len(data),1)

    print('x,  y shape',x.shape, y.shape)
    #for i in range(2000):
    i = 0
    err = 10
    while err>0.05:
        estimate = forward(x, weights_hidden, weights_output)
        dw0, dw1, loss = backward(y, x, weights_hidden, weights_output, estimate)
        weights_hidden, weights_output = gradientdescent(weights_hidden, weights_output, dw0, dw1)
        # print(new_w0.shape,new_w1.shape)
        #weights_hidden = new_w0
        #weights_output = new_w1
        err = np.mean(loss ** 2)
        if i % 100 == 0:
            print("Iter", i)
            print("Error", err)
            # print("Error", loss)
        i += 1
    #estimate[estimate >= 0] = 1
    #estimate[estimate < 0] = -1

    test = [1, 1.5, 1.4]
    pred = forward(test, weights_hidden, weights_output)
    pred[pred < 0] = -1
    pred[pred >= 0] = 1
    print(pred)
    plotdata(data)
    plot3d(data, estimate)

    # print(weights_output,weights_hidden)

if __name__ == '__main__':
    main()
