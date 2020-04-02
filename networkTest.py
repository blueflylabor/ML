import numpy as np 
import matplotlib.pyplot as plt 

# loaddata
def loaddata(filename):
    fr=open(filename)
    x=[]
    y=[]
    for line in fr.readlines():
        line=line.strip().split()
        x.append([float(line[0]),float(line[1])])
        y.append([float(line[-1])])
    return np.mat(x),np.mat(y)

# data scaling
def scaling(data):
    max=np.max(data,0)
    min=np.min(data,0)
    return(data-min)/(max-min),max,min

# sigmoid
def sigmoid(data):
    return 1/(1+np.exp(-data))

# w b calc
def wb_calc(X,ymat,alpha=0.1,maxIter=10000,n_hidden_dim=3,reg_lambda=0):
    # init w b
    W1 = np.mat(np.random.randn(2,n_hidden_dim))
    b1 = np.mat(np.random.randn(1,n_hidden_dim))
    W2 = np.mat(np.random.randn(n_hidden_dim, 1))
    b2 = np.mat(np.random.randn(1, 1))
    w1_save = []
    b1_save = []
    w2_save = []
    b2_save = []
    ss = []
    for stepi in range(maxIter):
        # FP
        z1 = X*W1 + b1 # (20,2)(2,3) + (1,3) = (20,3)
        a1 = sigmoid(z1) # (20,3)
        z2 = a1*W2 + b2 # (20,3)(3,1) + (1,1) = (20,1)
        a2 = sigmoid(z2) # (20,1)
        # BP
        a0= X.copy()
        delta2 = a2 - ymat # (20,1)
        delta1 = np.mat((delta2*W2.T).A * (a1.A*(1-a1).A))
        #              (20,1)(1,3) .* (20,3) = (20,3)
        dW1 = a0.T*delta1 + reg_lambda*W1 # (2,20)(20,3) + (2,3) = (2,3)
        db1 = np.sum(delta1,0)
        # db1 = np.mat(np.ones()) * delta1
        dW2 = a1.T*delta2 + reg_lambda*W2
        db2 = np.sum(delta2,0)
        # undate w b
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        return W1,b1,W2,b2
# implement 
xmat,ymat=loaddata('nn_data.txt')
#print(xmat,xmat.shape,ymat,ymat.shape)
xmat_s,xmat_max,xmat_min=scaling(xmat)
#print(xmat_s,xmat_s.shape)
W1,b1,W2,b2=wb_calc(xmat_s,ymat,0.05,20000,20,0)

# show
plotx1=np.arange(0,10,0.01)
plotx2=np.arange(0,10,0.01)
plotX1,plotX2=np.meshgrid(plotx1,plotx2)
plotx_new=np.c_[plotX1.ravel(),plotX2.ravel()]
plotx_new2=(plotx_new-xmat_min)/(xmat_max-xmat_min)
plot_z1=plotx_new*W1+b1
plot_a1=sigmoid(plot_z1)
plot_z2=plot_a1*W2+b2
plot_a2=sigmoid(plot_z2)
ploty_new=np.reshape(plot_a2,plotX1.shape)
plt.contourf(plotX1,plotX2,ploty_new,alpha=0.5)
plt.scatter(xmat[:,0][ymat==0].A,xmat[:,1][ymat==0].A,s=100,marker='o',label='0')
plt.scatter(xmat[:,0][ymat==1].A,xmat[:,1][ymat==1].A,s=100,marker='^',label='1')
plt.grid()
plt.legend()
plt.show()
