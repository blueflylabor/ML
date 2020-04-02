#可视化 x,y坐标系
import numpy as np
import matplotlib.pyplot as plt

#meshgrid
x=np.arange(0,11,1)
y=np.arange(0,11,1)
xx,yy=np.meshgrid(x,y)
#print("xx:",xx)
#print("yy:",yy)
zz=np.sin(xx)+np.cos(yy)
plt.contourf(xx,yy,zz,3)
cont=plt.contour(xx,yy,zz,3,colors='black')
plt.clabel(cont,inline=True,fontsize=10)
plt.show()