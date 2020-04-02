import numpy as np
import matplotlib.pyplot as plt 

def loaddata(filename):
    fr=open(filename)
    x=[]
    y=[]
    for line in fr.readlines():
        line=line.strip().split()
        x.append(float(line[0]))
        y.append(float(line[1]))
    xmat=np.mat(x).T
    ymat=np.mat(y).T
    return xmat,ymat




def wb_calc(xmat,ymat,lam=0,alpha=0.00001,maxIter=20000):
    m,n=xmat.shape
    X=np.mat(np.zeros((len(xmat),3)))
    X[:,0]=xmat
    X[:,1]=xmat.A**2
    X[:,2]=xmat.A**3

    W=np.mat(np.random.randn(3,1))
    b=np.mat(np.random.randn(1,1))
    '''
    print('X',X)
    print('W',W)
    print('b',b)
    '''
    
    W0=W.copy()
    b0=b.copy()
    #print('W0',W0)
    #print('b0',b0)
    for i in range(maxIter):
        H=X*W+b
        dw=1/m * X.T*(H-ymat)+1/m*lam*W
        db=1/m * np.sum(H-ymat)

        W-=alpha*dw
        b-=alpha*db
    #print('W',W)
    #print('b',b)
    return W,b,W0,b0
        

xmat,ymat=loaddata('test_data')
print("xmat:",xmat,xmat.shape,type(xmat))
print("ymat:",ymat,ymat.shape,type(ymat))
W,b,W0,b0=wb_calc(xmat,ymat,10000,0.00001,50000)

plotx=np.arange(0,10,0.001)
w1=W[0,0]
w2=W[1,0]
w3=W[2,0] 

ploth=w1*plotx+w2*plotx**2+w3*plotx**3+b[0,0]
plt.plot(plotx,ploth,label='h_model') 
w1_0=W0[0,0]
w2_0=W0[1,0]
w3_0=W0[2,0] 

ploth_0=w1_0*plotx+w2_0*plotx**2+w3_0*plotx**3+b[0,0]
plt.plot(plotx,ploth_0,label='h_init_model') 
plt.scatter(xmat.A,ymat.A,s=50,c='r')#转换维数组
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
