from scipy import ndimage
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from ex5_pca import pca

display=1
rows = 136
columns = 38880 #
imgshape =[243,160]
dir = 'yalefaces_cropBackground'

def readimages(folder):
    """ Read image data  """
    im = [ndimage.imread(folder+'/'+filename,flatten=True) for filename in os.listdir(folder)]
    X = np.asarray(im).reshape(rows,columns)
    return X

def displayimage(data,imgshape,k_clusters):
    """ display image """
    fig = plt.figure()
    i=0
    for val in range(k_clusters):
        i=i+1
        img1 = data[val].reshape(imgshape)
        fig.add_subplot(k_clusters,1,i)
        plt.imshow(img1, cmap='gray')
    plt.show()

def cal_error(x, centroid, r_nk, k):
    totalerror = 0
    for n in range(rows):  # 136
        tmp = [x[n] - centroid[i] for i in range(k)]
        tmp = np.asarray(tmp)
        error = np.linalg.norm(tmp, ord=2, axis=1)
        totalerror += np.matmul(r_nk[n,:],error)
    return totalerror

def k_means(data,k_clusters):
    opt_centroid = np.zeros([k_clusters, columns])
    opt_r_nk = np.zeros([rows, k_clusters])
    errorvector = np.zeros(10)

    for iter in range(10):
        count = 1
        mean = [data[i] for i in [random.randint(0, rows - 1) for j in range(k_clusters)]]
        mean = np.asarray(mean)
        #print("mean shape", mean.shape)
        newmean = np.zeros([k_clusters, columns])
        #error = np.empty([rows, columns])
        r = np.zeros([rows, k_clusters])

        while not np.array_equal(mean,newmean):
            if count!=1:
                mean = newmean.copy()
            #r = np.zeros([rows, k_clusters])
            for j in range(rows):
                tmp = [data[j] - mean[i] for i in range(k_clusters)]
                tmp = np.asarray(tmp)
                error = np.linalg.norm(tmp,ord=2,axis=1)
                #print(error)
                r[j][error.argmin()] = 1

            for i in range(k_clusters):
                denom = sum(r[j][i] for j in range(rows))
                num = sum(r[j][i] * data[j] for j in range(rows))
                if denom != 0:
                    newmean[i] = num / denom
            count += 1

        error=cal_error(data,newmean,r,k_clusters)
        #print(error)
        errorvector[iter] = error
        if iter == 0:
            opt_centroid = mean
            opt_r_nk = r
        elif error < errorvector[iter]:
            opt_centroid = mean
            opt_r_nk = r

    print("Centroid",opt_centroid)
    print("Error vector",errorvector)
    return opt_centroid, opt_r_nk,errorvector.min()

if __name__=='__main__':
    #k_array = [4, 7, 10, 15, 17, 20]
    apply_pca = True
    k_array = [4, 7]
    targetdim = 20

    k_error = []

    img = readimages(dir)
    print("data shape",img.shape)

    for k in k_array:
        if apply_pca:
            img = pca(img,imgshape,rows,columns,targetdim)
        center, r, error = k_means(img,k)
        k_error.append(error)
        displayimage(center,imgshape,k)

    print("Min errors",k_error)
    plt.plot(k_array,k_error)
    plt.xlabel("Values of K")
    plt.ylabel("Error")
    plt.show()
#displayimage(newmean,imgshape)

