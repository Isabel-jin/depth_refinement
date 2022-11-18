import numpy as np

# def ShadingGradients(B,I):
    # Here we still assume albedo = 1
    # Eg = np.zeros((Nx,Ny))
    # for i in range(0,Nx-1):
    #    for j in range(0,Ny-1):
    #        Eg[i][j] = ((B[i][j] - B[i+1][j])-(I[i][j]-I[i+1][j]))**2 + ((B[i][j] - B[i][j+1])-(I[i][j]-I[i][j+1]))**2
    
    # return Eg

def ShadingGradients1(B,I,n,wg):   #B:rgb图像 I:灰度图 n:点序数 wg:梯度约束权重
    Nx = B.shape(0)
    return pow(wg,0.5)* [B(n)-B(n+Nx)-(I(n)-I(n+Nx))]
    

def ShadingGradients2(B,I,n,wg):

    return pow(wg,0.5)* [B(n)-B(n+1)-(I(n)-I(n+1))]


# def SmoothConstraint(D):
  #  p = np.zeros(Nx*Ny*3).reshape(Nx,Ny,3)
   # for i in range(0,Nx):
    #    for j in range (0,Ny):
     #       cameraMatrix = np.array([(i-ux)/fx,(j-uy)/fy,1])
      #      p[i][j] = D[i][j] * cameraMatrix
    # Es = np.zeros((Nx,Ny))
    # for i in range(1,Nx-1):
    #    for j in range(1,Ny-1):  
    #        a = p[i][j] - ws*(p[i-1][j] + p[i][j-1]+p[i+1][j]+p[i][j+1])
    #        Es[i][j] = np.linalg.norm(a)**2

    #return Es 

def SmoothConstraint1(D,n,ws):
    i = n //Nx 
    j = n - i*Nx
    r3 = D(n) - ws*(D(n-Nx)+D(N-1)+D(n+Nx)+D(n+1))
    r1 = (i-ux)/fx *r3
    return pow(ws,0.5)*r1

def SmoothConstraint2(D,n,ws):
    i = n //Nx 
    j = n - i*Nx
    r3 = D(n) - ws*(D(n-Nx)+D(N-1)+D(n+Nx)+D(n+1))
    r2 = (j-uy)/fy *r3
    return pow(ws,0.5)*r2

def SmoothConstraint3(D,n,ws):
    r3 = D(n) - ws*(D(n-Nx)+D(N-1)+D(n+Nx)+D(n+1))
    return pow(ws,0.5)*r3

def DepthConstraint(D,Di,n,wp):
    
    return pow(wp,0.5)*(D(n) - Di(n)) 


def Jacobi():

    return 

def GaussNewton(Em,D):
      
    return Dnew