from PIL import Image
import numpy as np
import math
#相机矩阵
fx=975.9293
fy=975.9497
cx=1029.9879
cy=766.8806
#裁切参数
i1=700
i2=1160
j1=820
j2=1200
def image_generation(depth_image_path,image_path,out_path,fx,fy,cx,cy,i1,i2,j1,j2):
    D=np.array(Image.open(depth_image_path))
    Ic=Image.open(image_path)
    grey_im=Ic.convert('L')
    I=np.array(grey_im)
    I_cut=I[i1:i2,j1:j2]
    I_cut=Image.fromarray(I_cut)
    I_cut.convert('L').save(out_path+'I_cut.png')
    mask=np.zeros((D.shape[0],D.shape[1]))
    for i in range(i1,i2+1):
        for j in range(j1,j2+1):
                mask[i][j]=1
    I_RESHAPE=np.zeros((I.shape[0],I.shape[1]))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
                if mask[i][j]==0:
                    I_RESHAPE[i][j]=-1
                else :
                    I_RESHAPE[i][j]=I[i][j]
    I_RESHAPE=I_RESHAPE.reshape(I.shape[0]*I.shape[1],1)
    n=np.zeros((D.shape[0],D.shape[1],3))
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
                if mask[i][j]==1:
                    n[i][j][0]=(D[i][j-1]*(D[i][j]-D[i-1][j])/fy)
                    n[i][j][1]=(D[i-1][j]*(D[i][j]-D[i][j-1])/fx)
                    n[i][j][2]=n[i][j][0]*(cx-i)/fx+n[i][j][1]*(cy-j)/fy-D[i-1][j]*D[i][j-1]/fx/fy
                    m=math.sqrt(pow(n[i][j][0],2)+pow(n[i][j][1],2)+pow(n[i][j][2],2))
                    n[i][j][0]/=m
                    n[i][j][1]/=m
                    n[i][j][2]/=m
    H=np.zeros((I.shape[0],I.shape[1],9))
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
    H_0removed=H[~np.all(H==0,axis=1)]
    I_0removed=I_RESHAPE[~np.all(I_RESHAPE==-1,axis=1)]
    l=np.linalg.inv(H_0removed.T@H_0removed)@H_0removed.T@I_0removed
    np.savetxt(fname=out_path+'l.csv',X=l,fmt="%f",delimiter=",")
    Ig=np.zeros((I.shape[0],I.shape[1]))
    H=H.reshape((1536,2048,9))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]): 
            sum=0
            for k in range(9):
                sum=sum+l[k]*H[i][j][k]
            Ig[i][j]=sum
    res=np.zeros((I.shape[0],I.shape[1]))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]): 
            if mask[i][j]!=0:
                res[i][j]=abs(I[i][j]-Ig[i][j])
    Ig=Ig[i1:i2,j1:j2]
    IG=Image.fromarray(Ig)
    IG.convert('L').save(out_path+'I_G.png')
    res=res[i1:i2,j1:j2]
    Res=Image.fromarray(res)
    Res.convert('L').save(out_path+'Res.png')
image_generation('depth/0000000102.png','image/0000000102.jpg','out/',fx,fy,cx,cy,i1,i2,j1,j2)