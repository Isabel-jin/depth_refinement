import numpy as np

fx = 975.9293
fy = 975.9497
ux = 1029.9879
uy = 766.8806

# Nx为原图每行的像素个数，N为图像总的像素个数
Nx = 460
N = 460 * 380

def ShadingGradients1(B,I,n,wg):   #B:rgb图像 I:灰度图 n:点序数 wg:梯度约束权重
    Nx = B.shape(0)
    return pow(wg,0.5)* [B[n]-B[n+Nx]-(I[n]-I[n+Nx])]
    

def ShadingGradients2(B,I,n,wg):
    return pow(wg,0.5)* [B[n]-B[n+1]-(I[n]-I[n+1])]
    
def SmoothConstraint1(D,n,ws):
    i = n //Nx 
    r3 = D[n] - ws*(D[n-Nx]+D[N-1]+D[n+Nx]+D[n+1])
    r1 = (i-ux)/fx *r3
    return pow(ws,0.5)*r1

def SmoothConstraint2(D,n,ws):
    i = n //Nx 
    j = n - i*Nx
    r3 = D[n] - ws*(D[n-Nx]+D[n-1]+D[n+Nx]+D[n+1])
    r2 = (j-uy)/fy *r3
    return pow(ws,0.5)*r2

def SmoothConstraint3(D,n,ws):
    r3 = D[n] - ws*(D[n-Nx]+D[n-1]+D[n+Nx]+D[n+1])
    return pow(ws,0.5)*r3

def DepthConstraint(D,Di,n,wp):    
    return pow(wp,0.5)*(D[n] - Di[n]) 

def PartialNum1(D,n,delta):
    D_delta = D
    D_delta[n] = D[n] + delta
    B = D2B(D)
    B_delta = D2B(D_delta)
    R = ShadingGradients1(B,I,i,wg)
    R_delta = ShadingGradients1(B_delta,I,wg)
    return (R_delta - R)/delta

def PartialNum2(D,n,delta):
    D_delta = D
    D_delta[n] = D[n] + delta
    B = D2B(D)
    B_delta = D2B(D_delta)
    R = ShadingGradients2(B,I,i,wg)
    R_delta = ShadingGradients2(B_delta,I,wg)
    return (R_delta - R)/delta

def PartialNum3(D,n,delta):
    D_delta = D
    D_delta[n] = D[n] + delta
    R = SmoothConstraint1(D,n,ws)
    R_delta = SmoothConstraint1(D_delta,n,ws)
    return (R_delta - R)/delta

def PartialNum4(D,n,delta):
    D_delta = D
    D_delta[n] = D[n] + delta
    R = SmoothConstraint2(D,n,ws)
    R_delta = SmoothConstraint2(D_delta,n,ws)
    return (R_delta - R)/delta

def PartialNum5(D,n,delta):
    D_delta = D
    D_delta[n] = D[n] + delta
    R = SmoothConstraint3(D,n,ws)
    R_delta = SmoothConstraint3(D_delta,n,ws)
    return (R_delta - R)/delta

def PartialNum6(D,n,delta):
    return 1

def Jacobi(D):
    delta = 0.1
    J = np.zeros((6*N,N))
    for i in range(0,N):    #第一个N函数组:r_0-r_N-1
        n = i
        # pr/pD(n)
        J[i][n] = PartialNum1(D,n,delta)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum1(D,n-1,delta)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum1(D,n-Nx,delta)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum1(D,n+Nx,delta)
        # pr/pD(n+Nx-1)
        J[i][n+Nx-1] = PartialNum1(D,n+Nx-1,delta)
    for i in range(N,2*N):    #第二个N函数组:r_N-r_2N-1
        n = i -N
        # pr/pD(n)
        J[i][n] = PartialNum2(D,n,delta)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum2(D,n-1,delta)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum2(D,n-Nx,delta)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum2(D,n+1,delta)
        # pr/pD(n-Nx+1)
        J[i][n-Nx+1] = PartialNum2(D,n-Nx+1,delta)
    for i in range(2*N,3*N):    #第三个N函数组:r_2N-r_3N-1
        n = i-2*N
        # pr/pD(n)
        J[i][n] = PartialNum3(D,n,delta)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum3(D,n-1,delta)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum3(D,n-Nx,delta)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum3(D,n+1,delta)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum3(D,n+Nx,delta)
    for i in range(3*N,4*N):    #第三个N函数组:r_3N-r_4N-1
        n = i-3*N
        # pr/pD(n)
        J[i][n] = PartialNum4(D,n,delta)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum4(D,n-1,delta)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum4(D,n-Nx,delta)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum4(D,n+1,delta)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum4(D,n+Nx,delta)
    for i in range(4*N,5*N):    #第三个N函数组:r_4N-r_5N-1
        n = i-4*N
        # pr/pD(n)
        J[i][n] = PartialNum5(D,n,delta)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum5(D,n-1,delta)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum5(D,n-Nx,delta)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum5(D,n+1,delta)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum5(D,n+Nx,delta)
    for i in range(5*N,6*N):    #第六个函数组：r_5N-r_6N-1
        n = i-5*N
        J[i][n] = 1
    return J 


def GaussNewton(D):
    F = np.zeros((1,6*N))
    B = D2B(D)
    for i in range(0,N):
        F[1][i] = ShadingGradients1(B,I,i,wg) 
    for i in range(N,2*N):
        F[1][i] = ShadingGradients2(B,I,i-N,wg)
    for i in range(2*N,3*N):
        F[1][i] = SmoothConstraint1(D,i-2*N,ws)
    for i in range(3*N,4*N):
        F[1][i] = SmoothConstraint2(D,i-3*N,ws)
    for i in range(4*N,5*N):
        F[1][i] = SmoothConstraint3(D,i-4*N,ws)
    for i in range(5*N,6*N):
        F[1][i] = DepthConstraint(D,Di,i-5*N,wp)
    
    J = Jacobi(D)
    d = -np.matmul(np.matmul(np.linalg.inv(np.matmul(J.T,J)),J.T),F)
    Dnew = D + d
    return Dnew
