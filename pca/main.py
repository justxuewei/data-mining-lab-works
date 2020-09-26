import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def zeroMean(dataMat):
    """
    :param dataMat: each row in data_mat represents a sample, and each column represents the same characteristic.
    """
    # mean by column, i.e. get the mean of each characteristic.
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covmat = np.cov(newData, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    eigValInd = np.argsort(eigVals)
    n_eigValInd = eigValInd[: -(n+1): -1]
    n_eigVect = eigVects[:, n_eigValInd]
    lowDDataMat = newData * n_eigVect
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    return np.array(lowDDataMat), np.array(reconMat)


def loadDataSet(filename):
    df = pd.read_table(filename, sep="\t")
    return np.array(df)


def showData(dataMat, reconMat, subplot=111):
    fig = plt.figure()
    ax = fig.add_subplot(subplot)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c='green')
    ax.scatter(np.array(reconMat[:, 0]), reconMat[:, 1], c='red')
    plt.show()


if __name__ == "__main__":
    dataMat = loadDataSet("./testSet")
    lowDData, reconMat = pca(dataMat, 1)
    showData(dataMat, reconMat)
    newData, _ = zeroMean(dataMat)
    showData(dataMat, newData)
