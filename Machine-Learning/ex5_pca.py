import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
from scipy import ndimage
import os
import numpy as np
from random import randint



def readimages(folder):
    """
    Read image data
    """
    im = [ndimage.imread(folder+'/'+filename) for filename in os.listdir(folder)]
    X = np.asarray(im).reshape(rows,columns)
    imgshape = np.array(plt.imread(folder + '/' + 'subject01.gif')).shape
    return X, imgshape

def displayimage(data,data_recon,imgshape):
    """ display image"""
    fig = plt.figure()
    rands = [randint(0, rows-1) for p in range(samples)]
    #print(rands)
    i=0
    for val in rands:
        i=i+1
        img1 = data[val].reshape(imgshape)
        img2 = data_recon[val].reshape(imgshape)
        fig.add_subplot(gridx,gridy,i)
        plt.imshow(img1, cmap='gray')
        i=i+1
        fig.add_subplot(gridx,gridy,i)
        plt.imshow(img2, cmap='gray')
    plt.show()

def pca(data,shape,rows,columns, neigenvalues,isdisplay=False):

    print("PCA: data shape:",data.shape)
    #sp.sparse.linalg.svds(data,k=neigenvalues)
    mean = data.mean(0)
    one = np.ones([rows])
    Xt = data - one.reshape(rows,1) * mean.reshape(1,columns)
    #covar = np.matmul(Xt.T, Xt)
    u,s,v = sla.svds(Xt,neigenvalues)
    print("PCA: V shape",v.shape)

    z = np.matmul(Xt, v.T)
    #reconstruction
    data_recon = np.matmul(one.reshape(rows,1), mean.reshape(1,columns)) + np.matmul(z, v)

    #error
    error = data - data_recon
    error = np.linalg.norm(error,ord=2)

    if isdisplay:
        print("Error:", (error))
        displayimage(data, data_recon, shape)

    return data_recon

if __name__=='__main__':
    dir = 'yalefaces'
    rows = 166
    columns = 77760
    samples = 2
    gridx = samples
    gridy = 2
    targetdim = 60
    data, shape = readimages(dir)
    pca(data,shape,rows,columns,targetdim,isdisplay=True)

