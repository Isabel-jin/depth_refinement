from PIL import Image
import numpy as np
import cv2
import math
#获得深度图像
D=np.array(Image.open('RGBD_single/0_000000.png'))
#裁切参数
i1=370
i2=1240
j1=470
j2=1550
#灰度图像
Ic=Image.open('RGBD_single/0_000000.jpg')
grey_im=Ic.convert('L')
I=np.array(grey_im)
#对图片进行裁切
I_cut=I[i1:i2,j1:j2]
#用于计算L
I_RESHAPE=I.reshape(I.shape[0]*I.shape[1],1)
#法向量矩阵
n=np.zeros((D.shape[0],D.shape[1],3))
#球谐函数矩阵
H=np.zeros((I_cut.shape[0],I_cut.shape[1],9))
#相机内参
fx=974.20551
fy=973.9942
cx=1025.8949
cy=773.54004

#计算法向量矩阵n
for i in range(n.shape[0]):
    for j in range(n.shape[1]):
            if i==0 and j==0:
                n[i][j][0]=(D[i][j+1]*(D[i][j]-D[i+1][j])/fy)
                n[i][j][1]=(D[i+1][j]*(D[i][j]-D[i][j+1])/fx)
                n[i][j][2]=n[i][j][0]*(cx-i)/fx+n[i][j][1]*(cy-j)/fy-D[i+1][j]*D[i][j+1]/fx/fy
            elif i==0 and j!=0:
                n[i][j][0]=(D[i][j-1]*(D[i][j]-D[i+1][j])/fy)
                n[i][j][1]=(D[i+1][j]*(D[i][j]-D[i][j-1])/fx)
                n[i][j][2]=n[i][j][0]*(cx-i)/fx+n[i][j][1]*(cy-j)/fy-D[i+1][j]*D[i][j-1]/fx/fy
            elif i!=0 and j==0:
                 n[i][j][0]=(D[i][j+1]*(D[i][j]-D[i-1][j])/fy)
                 n[i][j][1]=(D[i-1][j]*(D[i][j]-D[i][j+1])/fx)
                 n[i][j][2]=n[i][j][0]*(cx-i)/fx+n[i][j][1]*(cy-j)/fy-D[i-1][j]*D[i][j+1]/fx/fy
            else : 
                 n[i][j][0]=(D[i][j-1]*(D[i][j]-D[i-1][j])/fy)
                 n[i][j][1]=(D[i-1][j]*(D[i][j]-D[i][j-1])/fx)
                 n[i][j][2]=n[i][j][0]*(cx-i)/fx+n[i][j][1]*(cy-j)/fy-D[i-1][j]*D[i][j-1]/fx/fy
            m=math.sqrt(pow(n[i][j][0],2)+pow(n[i][j][1],2)+pow(n[i][j][2],2))
            if m!=0:
                n[i][j][0]/=m
                n[i][j][1]/=m
                n[i][j][2]/=m
#计算用于得到l的H矩阵
#n= np.load(file="n.npy")
for i in range(I_cut.shape[0]):
    for j in range(I_cut.shape[1]):
        H[i][j][0]=1
        H[i][j][1]=n[i+i1][j+j1][1]
        H[i][j][2]=n[i+i1][j+j1][2]
        H[i][j][3]=n[i+i1][j+j1][0]
        H[i][j][4]=n[i+i1][j+j1][0]*n[i+i1][j+j1][1]
        H[i][j][5]=n[i+i1][j+j1][1]*n[i+i1][j+j1][2]
        H[i][j][6]=-pow(n[i+i1][j+j1][0],2)-pow(n[i+i1][j+j1][1],2)+2*pow(n[i+i1][j+j1][2],2)
        H[i][j][7]=n[i+i1][j+j1][2]*n[i+i1][j+j1][0]
        H[i][j][8]=pow(n[i+i1][j+j1][0],2)-pow(n[i+i1][j+j1][1],2)
H=np.reshape(H,(-1,9))

#计算lk
l=np.linalg.inv(H.T@H)@H.T@I_RESHAPE
print(l)
#保存数据
'''
np.savetxt(fname="D.csv",X=D,fmt="%f",delimiter=",")
np.save(file="n.npy",arr=n)
np.savetxt(fname="l.csv",X=l,fmt="%f",delimiter=",")
np.savetxt(fname="H.csv",X=H,fmt="%f",delimiter=",")
'''




        







