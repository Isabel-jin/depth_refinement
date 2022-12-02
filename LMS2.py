import numpy as np
def init(D,n):
    nx = D[n-1]*(D[n]-D[n-Nx])/fy 
    ny = D[n-Nx]*(D[n]-D[n-1])/fx 

    i = n // Nx
    j =  n - Nx * i
    a = (ux-i)/fx
    b = (uy-j)/fy

    nz = a*nx/fx + b*ny/fy - D[n-Nx]*D[n-1]/(fx*fy)

    dnx_dD1 = D[n-1]/fy     #dnx/dD(i,j)
    dnx_dD2 = (D[n]-D[n-Nx])/fy  #dnx/dD(i,j-1)
    dnx_dD3 = -D[n-1]/fy  #dnx/dD(i-1,j)

    dny_dD1 = D[n-Nx]/fx 
    dny_dD2 = -D[n-Nx]/fx 
    dny_dD3 = (D[n]-D[n-1])/fx 


    dnz_dD1 = a * dnx_dD1 + b * dny_dD1
    dnz_dD2 = a * dnx_dD2 + b * dny_dD2
    dnz_dD3 = a * dnx_dD3 + b * dny_dD3
    #dH_k(i,j)/dD(i,j)
    dh_dD1 = np.zeros(9)
    dh_dD1[0] = 0
    dh_dD1[1] = dny_dD1
    dh_dD1[2] = dnz_dD1
    dh_dD1[3] = dnx_dD1
    dh_dD1[4] = nx * dny_dD1 + ny * dnx_dD1
    dh_dD1[5] = ny * dnz_dD1 + nz * dny_dD1
    dh_dD1[6] = -2 * nx * dnx_dD1 - 2 * ny * dny_dD1 + 4 * nz * dnz_dD1
    dh_dD1[7] = nz * dnx_dD1 + nx * dnz_dD1
    dh_dD1[8] = 2 * nx * dnx_dD1 - 2 * ny * dny_dD1

    #dH_k(i,j)/dD(i,j-1)
    dh_dD2 = np.zeros(9)
    dh_dD2[0] = 0
    dh_dD2[1] = dny_dD2
    dh_dD2[2] = dnz_dD2
    dh_dD2[3] = dnx_dD2
    dh_dD2[4] = nx * dny_dD2 + ny * dnx_dD2
    dh_dD2[5] = ny * dnz_dD2 + nz * dny_dD2
    dh_dD2[6] = -2 * nx * dnx_dD2 - 2 * ny * dny_dD2 + 4 * nz * dnz_dD2
    dh_dD2[7] = nz * dnx_dD2 + nx * dnz_dD2
    dh_dD2[8] = 2 * nx * dnx_dD2 - 2 * ny * dny_dD2

    #dH_k(i,j)/dD(i-1,j)
    dh_dD3 = np.zeros(9)
    dh_dD3[0] = 0
    dh_dD3[1] = dny_dD3
    dh_dD3[2] = dnz_dD3
    dh_dD3[3] = dnx_dD3
    dh_dD3[4] = nx * dny_dD3 + ny * dnx_dD3
    dh_dD3[5] = ny * dnz_dD3 + nz * dny_dD3
    dh_dD3[6] = -2 * nx * dnx_dD3 - 2 * ny * dny_dD3 + 4 * nz * dnz_dD3
    dh_dD3[7] = nz * dnx_dD3 + nx * dnz_dD3
    dh_dD3[8] = 2 * nx * dnx_dD3 - 2 * ny * dny_dD3

def PartialB(n_B,n_D,l):
    r = 0
    init(D,n_B)
    if(n_B == n_D):
        for i in range(0,9):
           r = r + l[i] * dh_dD1[i]
    if(n_B == n_D + 1):
        for i in range(0,9):
           r = r + l[i] * dh_dD2[i]
    if(n_B == n_D + Nx):
        for i in range(0,9):
           r = r + l[i] * dh_dD3[i] 
    return r

def PartialNum1(n_E,n_D,wg,l):
    return pow(wg,0.5) * (PartialB(n_E,n_D,l) - PartialB(n_E+Nx,n_D,l))
def PartialNum2(n_E,n_D,wg,l):
    return pow(wg,0.5) * (PartialB(n_E,n_D,l) - PartialB(n_E+1,n_D,l)) 


def PartialNum3(n_E,n_D,ws):
    i = n_D // Nx
    a = (i-ux)/fx
    if(n_E == n_D + Nx):
        return -ws * a
    return a

def PartialNum4(n_E,n_D,ws):
    i = n_D // Nx
    j = n_D - Nx * i
    b =(j - uy)/fy
    if(n_E == n_D + Nx):
        return -ws * b
    return b

def PartialNum5(n_E,n_D,ws):
    if(n_E == n_D + Nx):
        return -ws
    return 1

def PartialNum6(D,n):
    return 1

def Jacobi(D):
    delta = 0.1
    J = np.zeros((6*N,N))
    for i in range(0,N):    #第一个N函数组:r_0-r_N-1
        n = i
        # pr/pD(n)
        J[i][n] = PartialNum1(n,n,wg,l)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum1(n,n-1,wg,l)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum1(n,n-Nx,wg,l)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum1(n,n+Nx,wg,l)
        # pr/pD(n+Nx-1)
        J[i][n+Nx-1] = PartialNum1(n,n+Nx-1,wg,l)
    for i in range(N,2*N):    #第二个N函数组:r_N-r_2N-1
        n = i -N
        # pr/pD(n)
        J[i][n] = PartialNum2(n,n,wg,l)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum2(n,n-1,wg,l)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum2(n,n-Nx,wg,l)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum2(n,n+1,wg,l)
        # pr/pD(n-Nx+1)
        J[i][n-Nx+1] = PartialNum2(n,n-Nx+1,l)
    for i in range(2*N,3*N):    #第三个N函数组:r_2N-r_3N-1
        n = i-2*N
        # pr/pD(n)
        J[i][n] = PartialNum3(n,n,ws)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum3(n,n-1,ws)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum3(n,n-Nx,ws)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum3(n,n+1,ws)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum3(n,n+Nx,ws)
    for i in range(3*N,4*N):    #第四个N函数组:r_3N-r_4N-1
        n = i-3*N
        # pr/pD(n)
        J[i][n] = PartialNum4(n,n,ws)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum4(n,n-1,ws)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum4(n,n-Nx,ws)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum4(n,n+1,ws)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum4(n,n+Nx,ws)
    for i in range(4*N,5*N):    #第五个N函数组:r_4N-r_5N-1
        n = i-4*N
        # pr/pD(n)
        J[i][n] = PartialNum5(n,n,ws)
        # pr/pD(n-1)
        J[i][n-1] = PartialNum5(n,n-1,ws)
        # pr/pD(n-Nx)
        J[i][n-Nx] = PartialNum5(n,n-Nx,ws)
        # pr/pD(n+1)
        J[i][n+1] = PartialNum5(n,n+1,ws)
        # pr/pD(n+Nx)
        J[i][n+Nx] = PartialNum5(n,n+Nx,ws)
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