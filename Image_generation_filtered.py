from PIL import Image
import numpy as np
#获得深度图像
D=np.array(Image.open('D_filtered.png'))
#获得mask
mask=np.zeros((D.shape[0],D.shape[1]))
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if D[i][j]!=0 and D[i-1][j]!=0 and D[i][j-1]!=0:
            mask[i][j]=1
#灰色图像
Im=Image.open('RGBD_single/0_000000.jpg')
grey_im=Im.convert('L')
I=np.array(grey_im)
for i in range(I.shape[0]):
    for j in range(I.shape[1]): 
        if mask[i][j]==0:
            I[i][j]=0
H=np.loadtxt(fname="H_filtered.csv",dtype=np.float32,delimiter=",")
l=np.loadtxt(fname="l_filtered.csv",dtype=np.float32,delimiter=",")
Ig=np.zeros((I.shape[0],I.shape[1]))
H=H.reshape((1536,2048,9))
#生成图像
for i in range(I.shape[0]):
    for j in range(I.shape[1]): 
        sum=0
        for k in range(9):
            sum=sum+l[k]*H[i][j][k]
        Ig[i][j]=sum
IG=Image.fromarray(Ig)
#np.savetxt(fname="grey.csv",X=I,fmt="%f",delimiter=",")
#np.savetxt(fname="Ig.csv",X=Ig,fmt="%f",delimiter=",")
IG.convert('L').save('I_G_filtered.png')
res=np.zeros((I.shape[0],I.shape[1]))

for i in range(I.shape[0]):
    for j in range(I.shape[1]): 
        if mask[i][j]!=0:
            res[i][j]=abs(I[i][j]-Ig[i][j])
np.savetxt(fname="res_filtered.csv",X=res,fmt="%f",delimiter=",")
Res=Image.fromarray(res)
Res.convert('L').save('Res_filtered.png')




