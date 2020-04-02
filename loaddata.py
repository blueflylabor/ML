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

xmat,ymat=loaddata('regression_data.txt')
print("xmat:",xmat,xmat.shape,type(xmat))
print("ymat:",ymat,ymat.shape,type(ymat))

plt.scatter(xmat.A,ymat.A,s=50,c='r')#转换维数组
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()