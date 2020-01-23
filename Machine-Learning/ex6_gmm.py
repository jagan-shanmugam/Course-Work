import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import random
import scipy.stats as sp
from matplotlib.patches import Ellipse

data = np.loadtxt("mixture.txt")
n = data.shape[0]
d = data.shape[1]
k = 3


def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(abs(vals))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def show(X, mu, cov):

    plt.cla()
    K = len(mu)  # number of clusters
    colors = ['r','r','r', 'k', 'g', 'c', 'm', 'y', 'r']
    plt.plot(X[:,0], X[:,1], 'r*')
    for k in range(K):
        plot_ellipse(mu[k], cov[k], alpha=0.6,color=colors[k % len(colors)])

def gmm(assignpos=False,display=False):

    pi = [1/k for i in range(k)]
    mean = np.asarray([data[i] for i in [random.randint(0, n - 1) for j in range(k)]])
    covariance = np.asarray([np.eye(2) for i in range(k)])
    posterior = np.empty([n,k])
    X = data
    print("data.shape:", data.shape)
    #P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) * \
    #             np.exp(-.5 * np.einsum('ij, ij -> i', X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))
    prob = lambda mu,cov,n, i: sp.multivariate_normal.pdf(data[n], mu[i], cov[i])
    iter = 1
    while(True):
        if not assignpos:
            for num in range(n):
                denom = sum(pi[j] * prob(mean,covariance,num, j) for j in range(k))
                numer = np.asarray([pi[i] * prob(mean,covariance,num,i) for i in range(k)])
                posterior[num,:] = (numer / denom)
        elif assignpos:
            for num in range(n):
                posterior[num,:] = np.eye(1,k,random.randint(0,k-1))
        #print(posterior)
        nk = posterior.sum(axis=0)
        #print(nk)
        pi = [nk[i]/n for i in range(k)]
        #print(pi)
        newmean = np.asarray([sum(posterior[i][j] * data[i] for i in range(n)) /nk[j] for j in range(k)])

        newcovariance = np.empty(covariance.shape)
        for j in range(k):
        #  for i in range(n):
        #      tmp = data[i]-newmean[j]
        #      tmp = tmp.reshape(2,1)
             #print(tmp)
        #      newcovariance[j] += (posterior[i][j] * np.matmul(tmp,tmp.T) / nk[j])
            x_mu = np.matrix(X - newmean[j])
            newcovariance[j] = np.array(1 / nk[j] * np.dot(np.multiply(x_mu.T, posterior[:, j]), x_mu))

        if np.linalg.norm(mean - newmean,np.inf)<1e-4:
            break
        mean = newmean
        covariance = newcovariance

        iter += 1
        assignpos = False
        if display and iter%5==0:
            print("current iter",iter)
            show(X, mean, covariance)
            plt.show()
    #if not display:
    print('Total iter:',iter)
    show(X, mean, covariance)
    plt.show()
    print(mean)
    print(covariance)

def plot_gmm(x,y,mean,covariance):
    dist = lambda n, i: sp.multivariate_normal.pdf(data[n], mean[i], covariance[i])
    z = np.asarray([dist(j,2) for j in range(n)])
    print(z)
    #plt.contour(x,y,z)
    xi = np.linspace(-4, 2, 300)
    yi = np.linspace(-4, 3, 300)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    plt.contour(xi, yi, zi, len(levels), linewidths=0.5, colors='k', levels=levels)
    plt.show()

#x = np.random.uniform(-2, 2, 300)
#y = np.random.uniform(-2, 2, 300)
#x = data[:, 0]
#y = data[:, 1]

if __name__=='__main__':
    gmm(assignpos=False,display=True)
    """
    fig = plt.figure(figsize=(13, 6))
    show(X, mean, covariance)
    plt.show()
    """
#plot_gmm(x,y,mean,covariance)
#plt.show()
#plt.plot(data[:,0],data[:,1],'ro')

#If we assign random mean and covariance first, rate of convergence depends on the values assigned to mean points and covariance
#If we assign posterior first and calculate parameters later, rate of convergence seems constant
