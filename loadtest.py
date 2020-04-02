# import numpy, matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
#from numpy import  *
#np.seterr(divide='ignore',invalid='ignore')

# def loaddata
def loaddata(filename):
    fr = open(filename)
    x = []
    y = []
    for line in fr.readlines():
        line = line.strip().split()
        x.append(float(line[0]))
        y.append(float(line[1]))
    xmat = np.mat(x).T
    ymat = np.mat(y).T
    print("xmat:",xmat)
    print("ymat:",ymat)
    return xmat, ymat

def wb_calc(xmat,ymat):
    global m,n,lam,alpha,X,W,b,maxIter
    maxIter=20000
    m,n=xmat.shape #num of samples
    lam=0
    alpha=0.001
    X=np.mat(np.zeros((4,3)))
    print("*"*30)
    print("init:num of samples:"+str(m)+"str(lamda):"+str(lam)+" learning rate:"+str(alpha))
    X[:,0]=xmat
    X[:,1]=xmat.A**2#:
    X[:,2]=xmat.A**3 

    #init w,b
    W=np.random.randn(3,1)
    b=np.random.randn(1,1)
    W0=W.copy()
    b0=b.copy()
    print("*"*30)
    print("W0:",W0)
    print("b0",b)
    for i in range(maxIter):
        # dw, db
        H = X*W+b
        dw = 1/m * X.T*(H-ymat) + 1/m * lam*W
        #           (3,4)(4,1) + (3,1) = (3,1)
        db = 1/m * np.sum(H-ymat) #(1,1)

        # w,b update
        W -= alpha * dw
        b -= alpha * db
    print("*"*30)
    print("W:",W)
    print("b",b)
#show
xmat,ymat=loaddata('regression_data.txt')
wb_calc(xmat,ymat)