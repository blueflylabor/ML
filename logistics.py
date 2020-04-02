import numpy as np
import math
import matplotlib.pyplot as plt 
import pandas as pd


dataSet=pd.read_table('testSet.txt',header=None,sep='\t')
dataSet.columns=['X1','X2','labels']
dataSet.head()

def sigmoid(inX):
    return (1/(1+np.exp(-inX)))


def regularize(xMat):
    inMat=xMat.copy()
    inMeans=np.mean(xMat,axis=0)
    invar=np.std(inMat,axis=0)
    inMat=(inMat-inMeans)/invar
    return inMat


def BGD_LR(dataSet,alpha=0.001,maxCycles=500):
    xMat=np.mat(dataSet.iloc[:,:-1].values)
    yMat=np.mat(dataSet.iloc[:,-1].values).T
    xMat=regularize(xMat)
    m,n=xMat.shape
    weights=np.zeros((n,1))
    for i in range(maxCycles):
        grad=xMat.T*(xMat*weights-yMat)/m
        weights=weights-alpha*grad
    return weights


def SGD_LR(dataSet,alpha=0.001,maxCycles=500):
    dataSet = dataSet.sample(maxCycles, replace=True)
    dataSet.index = range(dataSet.shape[0])
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = regularize(xMat)
    m, n = xMat.shape
    weights = np.zeros((n,1))
    for i in range(m):
        grad = xMat[i].T * (xMat[i] * weights - yMat[i])
        weights = weights - alpha * grad
    return weights


def logisticAcc(dataSet, method, alpha=0.01, maxCycles=500):
    weights = method(dataSet,alpha=alpha,maxCycles=maxCycles)
    p = sigmoid(xMat * weights).A.flatten()
    for i, j in enumerate(p):
        if j < 0.5:
            p[i] = 0
        else:
            p[i] = 1
    train_error = (np.fabs(yMat.A.flatten() - p)).sum()
    trainAcc = 1 - train_error / yMat.shape[0]
    return trainAcc

xMat=np.mat(dataSet.iloc[:,:-1].values)
yMat=np.mat(dataSet.iloc[:,-1].values).T
m,n=xMat.shape
weights=np.zeros((m,1))
w=BGD_LR(dataSet)
xMat=regularize(xMat)
train=logisticAcc(dataSet,BGD_LR)
print(train)

