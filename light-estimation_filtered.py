from PIL import Image
import numpy as np
import cv2
import math

#获得深度图像
D=np.array(Image.open('D_filtered.png'))
#获得mask
mask=np.zeros((D.shape[0],D.shape[1]))
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if D[i][j]!=0 and D[i-1][j]!=0 and D[i][j-1]!=0:
            mask[i][j]=1
#灰度原始图像
Ic=Image.open('RGBD_single/0_000000.jpg')
grey_im=Ic.convert('L')
I=np.array(grey_im)
#用于计算L
I_RESHAPE=np.zeros((I.shape[0],I.shape[1]))
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
            if mask[i][j]==0:
                I_RESHAPE[i][j]=-1
            else :
                I_RESHAPE[i][j]=I[i][j]
I_RESHAPE=I_RESHAPE.reshape(I.shape[0]*I.shape[1],1)
#法向量矩阵
n=np.zeros((D.shape[0],D.shape[1],3))
#球谐函数矩阵
H=np.zeros((I.shape[0],I.shape[1],9))
#相机内参
fx=974.20551
fy=973.9942
cx=1025.8949
cy=773.54004
#计算法向量矩阵n
for i in range(n.shape[0]):
    for j in range(n.shape[1]):
            if mask[i][j]!=0:
                 n[i][j][0]=(D[i][j-1]*(D[i][j]-D[i-1][j])/fy)
                 n[i][j][1]=(D[i-1][j]*(D[i][j]-D[i][j-1])/fx)
                 n[i][j][2]=n[i][j][0]*(cx-i)/fx+n[i][j][1]*(cy-j)/fy-D[i-1][j]*D[i][j-1]/fx/fy
                 m=math.sqrt(pow(n[i][j][0],2)+pow(n[i][j][1],2)+pow(n[i][j][2],2))
                 n[i][j][0]/=m
                 n[i][j][1]/=m
                 n[i][j][2]/=m
#生成法向量图

I_n=np.zeros((D.shape[0],D.shape[1],3))
I_n=n*255
I_n=Image.fromarray(np.uint8(I_n))
I_n.show()
I_n.save('I_n_filtered.jpg')
#计算用于得到l的H矩阵
#n= np.load(file="n.npy")
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        if mask[i][j]!=0 :
            H[i][j][0]=1
            H[i][j][1]=n[i][j][1]
            H[i][j][2]=n[i][j][2]
            H[i][j][3]=n[i][j][0]
            H[i][j][4]=n[i][j][0]*n[i][j][1]
            H[i][j][5]=n[i][j][1]*n[i][j][2]
            H[i][j][6]=-pow(n[i][j][0],2)-pow(n[i][j][1],2)+2*pow(n[i][j][2],2)
            H[i][j][7]=n[i][j][2]*n[i][j][0]
            H[i][j][8]=pow(n[i][j][0],2)-pow(n[i][j][1],2)
H=np.reshape(H,(-1,9))
#去除零行
H_0removed=H[~np.all(H==0,axis=1)]
I_0removed=I_RESHAPE[~np.all(I_RESHAPE==-1,axis=1)]
#计算lk
l=np.linalg.inv(H_0removed.T@H_0removed)@H_0removed.T@I_0removed

#保存数据
#np.savetxt(fname="D.csv",X=D,fmt="%f",delimiter=",")
#np.save(file="n.npy",arr=n)
np.savetxt(fname="l_filtered.csv",X=l,fmt="%f",delimiter=",")
np.savetxt(fname="H_filtered.csv",X=H,fmt="%f",delimiter=",")





        







