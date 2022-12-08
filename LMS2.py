import numpy as np
import constrains
import InvLU

# 相机内参
fx = 975.9293
fy = 975.9497
ux = 1029.9879
uy = 766.8806

# Nx为原图每行的像素个数，N为图像总的像素个数
Nx = 154
N = 154 * 127

l = [51.040569,
     -11.897069,
     -78.228003,
     -18.870133,
     0.716704,
     -34.047397,
     -9.706077,
     6.701684,
     0.406902]


def PartialHk(D, n):
    '''
    对深度数组的每一位初始化，得到Hk对D的偏导
    :param D: 一维数组，记录每个像素点的深度
    :param n: 深度数组的索引
    :return dh_dD: Hk对D的偏导
    '''
    if n-1 < 0 or n-1 >= N:
        return np.zeros((3, 9))
    if n-Nx < 0 or n-Nx >= N:
        return np.zeros((3, 9))

    # 分别求第n个点的x,y,z方向法向量
    nx = D[n-1]*(D[n]-D[n-Nx])/fy 
    ny = D[n-Nx]*(D[n]-D[n-1])/fx 

    i = n // Nx
    j = n - Nx * i
    a = (ux-i)/fx
    b = (uy-j)/fy

    nz = a*nx/fx + b*ny/fy - D[n-Nx]*D[n-1]/(fx*fy)

    # 该点三个方向的法向量对该点和近邻点的偏导
    dnx_dD1 = D[n-1]/fy     #dnx/dD(i,j)
    dnx_dD2 = (D[n]-D[n-Nx])/fy  #dnx/dD(i,j-1)
    dnx_dD3 = -D[n-1]/fy  #dnx/dD(i-1,j)

    dny_dD1 = D[n-Nx]/fx 
    dny_dD2 = -D[n-Nx]/fx 
    dny_dD3 = (D[n]-D[n-1])/fx

    dnz_dD1 = a * dnx_dD1 + b * dny_dD1
    dnz_dD2 = a * dnx_dD2 + b * dny_dD2
    dnz_dD3 = a * dnx_dD3 + b * dny_dD3


    # 该点二阶九项球谐系数Hk对该点和近邻点深度求偏导
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

    dh_dD = np.zeros((3, 9))
    dh_dD[0, :] = dh_dD1
    dh_dD[1, :] = dh_dD2
    dh_dD[2, :] = dh_dD3
    return dh_dD


def PartialB(D, n_B, n_D, l):
    '''
    B对D的偏导
    :param D:一维深度数组
    :param n_B: B的索引
    :param n_D: D的索引
    :param l: 球谐系数
    :return: 返回相应的偏导值
    '''
    r = 0
    # 光照系数对深度图的偏导数
    dh_dD = PartialHk(D, n_B)
    if(n_B == n_D):
        for i in range(0,9):
           r = r + l[i] * dh_dD[0][i]
    if(n_B == n_D + 1):
        for i in range(0,9):
           r = r + l[i] * dh_dD[1][i]
    if(n_B == n_D + Nx):
        for i in range(0,9):
           r = r + l[i] * dh_dD[2][i]
    return r


def PartialNum1(D, n_E, n_D, wg, l):
    '''
    阴影梯度约束 Eg对的第一项偏导
    :param D: 一维深度数组
    :param n_E: Eg的索引
    :param n_D: D的索引
    :param wg: 超参数
    :param l: 二阶九项光照系数
    :return: 偏导值
    '''
    if n_E+Nx < 0 or n_E+Nx >= N:
        return 0
    return pow(wg, 0.5) * (PartialB(D, n_E, n_D, l) - PartialB(D, n_E+Nx, n_D, l))

def PartialNum2(D,n_E,n_D,wg,l):
    '''
    阴影梯度约束 Eg的第二项偏导
    :param D: 一维深度数组
    :param n_E: Eg的索引
    :param n_D: D的索引
    :param wg: 超参数
    :param l: 二阶九项光照系数
    :return: 偏导值
    '''
    if n_E+1 < 0 or n_E+1 >= N:
        return 0
    return pow(wg, 0.5) * (PartialB(D, n_E, n_D, l) - PartialB(D, n_E+1, n_D, l))

def PartialNum3(n_E,n_D,ws):
    '''
    平滑约束的第一项偏导
    :param n_E: Es的索引
    :param n_D: D的索引
    :param ws: 超参数
    :return: 偏导值
    '''
    i = n_D // Nx
    a = (i-ux)/fx
    if(n_E == n_D + Nx):
        return -ws * a
    return a

def PartialNum4(n_E,n_D,ws):
    '''
    平滑约束的第二项偏导
    :param n_E: Es的索引
    :param n_D: D的索引
    :param ws: 超参数
    :return: 偏导值
    '''
    i = n_D // Nx
    j = n_D - Nx * i
    b =(j - uy)/fy
    if(n_E == n_D + Nx):
        return -ws * b
    return b

def PartialNum5(n_E,n_D,ws):
    '''
    平滑约束的第三项偏导
    :param n_E: Es的索引
    :param n_D: D的索引
    :param ws: 超参数
    :return: 偏导值
    '''
    if(n_E == n_D + Nx):
        return -ws
    return 1

def PartialNum6(D,n):
    #深度约束的偏导
    return 1

def Jacobi(D,w,l):
    '''
    构造 6N * N的雅可比矩阵
    :param D: 一维深度数组
    :param w: 超参数
    :param l: 九项光照系数
    :return: 计算好的雅可比矩阵
    '''
    J = np.zeros((6*N, N), np.float16)
    wg = int(w[0])
    ws = int(w[1])
    # 遍历N个像素点，求6N项雅可比
    # 第一个N函数组:r_0-r_N-1 阴影梯度约束第一项
    for i in range(0, N):
        n = i
        # pr/pD(n)
        J[i][n] = PartialNum1(D, n, n, wg, l)
        # pr/pD(n-1)
        if n-1 >= 0 and n-1 < N:
            J[i][n-1] = PartialNum1(D, n, n-1, wg, l)
        # pr/pD(n-Nx)
        if n-Nx >= 0 and n-Nx < N:
            J[i][n-Nx] = PartialNum1(D, n, n-Nx, wg, l)
        # pr/pD(n+Nx)
        if n+Nx >= 0 and n+Nx < N:
            J[i][n+Nx] = PartialNum1(D, n, n+Nx, wg, l)
        # pr/pD(n+Nx-1)
        if n+Nx-1 >= 0 and n+Nx-1 < N:
            J[i][n+Nx-1] = PartialNum1(D, n, n+Nx-1, wg, l)
    print("J_1 finished!")

    # 第二个N函数组:r_N-r_2N-1 阴影梯度约束第二项
    for i in range(N, 2*N):
        n = i - N
        # pr/pD(n)
        J[i][n] = PartialNum2(D, n, n, wg, l)
        # pr/pD(n-1)
        if n-1 >= 0 and n-1 < N:
            J[i][n-1] = PartialNum2(D, n, n-1, wg, l)
        # pr/pD(n-Nx)
        if n-Nx >= 0 and n-Nx < N:
            J[i][n-Nx] = PartialNum2(D, n, n-Nx, wg, l)
        # pr/pD(n+1)
        if n+1 >= 0 and n+1 < N:
            J[i][n+1] = PartialNum2(D, n, n+1, wg, l)
        # pr/pD(n-Nx+1)
        if n-Nx+1 >= 0 and n-Nx+1 < N:
            J[i][n-Nx+1] = PartialNum2(D, n, n-Nx+1, wg, l)
    print("J_2 finished!")

    # 第三个N函数组:r_2N-r_3N-1  平滑约束第一项
    for i in range(2*N, 3*N):
        n = i-2*N
        # pr/pD(n)
        J[i][n] = PartialNum3(n, n, ws)
        # pr/pD(n-1)
        if n-1 >= 0 and n-1 < N:
            J[i][n-1] = PartialNum3(n, n-1, ws)
        # pr/pD(n-Nx)
        if n-Nx >= 0 and n-Nx < N:
            J[i][n-Nx] = PartialNum3(n, n-Nx, ws)
        # pr/pD(n+1)
        if n+1 >= 0 and n+1 < N:
            J[i][n+1] = PartialNum3(n, n+1, ws)
        # pr/pD(n+Nx)
        if n+Nx >= 0 and n+Nx < N:
            J[i][n+Nx] = PartialNum3(n, n+Nx, ws)
    print("J_3 finished!")

    # 第四个N函数组:r_3N-r_4N-1  平滑约束第二项
    for i in range(3*N, 4*N):
        n = i-3*N
        # pr/pD(n)
        J[i][n] = PartialNum4(n, n, ws)
        # pr/pD(n-1)
        if n-1 >= 0 and n-1 < N:
            J[i][n-1] = PartialNum4(n, n-1, ws)
        # pr/pD(n-Nx)
        if n-Nx >= 0 and n-Nx < N:
            J[i][n-Nx] = PartialNum4(n, n-Nx, ws)
        # pr/pD(n+1)
        if n+1 >= 0 and n+1 < N:
            J[i][n+1] = PartialNum4(n, n+1, ws)
        # pr/pD(n+Nx)
        if n+Nx >= 0 and n+Nx < N:
            J[i][n+Nx] = PartialNum4(n, n+Nx, ws)
    print("J_4 finished!")

    # 第五个N函数组:r_4N-r_5N-1  平滑约束第三项
    for i in range(4*N, 5*N):
        n = i-4*N
        # pr/pD(n)
        J[i][n] = PartialNum5(n, n, ws)
        # pr/pD(n-1)
        if n-1 >= 0 and n-1 < N:
            J[i][n-1] = PartialNum5(n, n-1, ws)
        # pr/pD(n-Nx)
        if n-Nx >= 0 and n-Nx < N:
            J[i][n-Nx] = PartialNum5(n, n-Nx, ws)
        # pr/pD(n+1)
        if n+1 >= 0 and n+1 < N:
            J[i][n+1] = PartialNum5(n, n+1, ws)
        # pr/pD(n+Nx)
        if n+Nx >= 0 and n+Nx < N:
            J[i][n+Nx] = PartialNum5(n, n+Nx, ws)
    print("J_5 finished!")

    # 第六个函数组：r_5N-r_6N-1 深度约束
    for i in range(5*N, 6*N):
        n = i-5*N
        J[i][n] = 1
    print("J_6 finished!")

    return J 


def GaussNewton(F, B, Di, I, D, w):
    '''
    :param F: 待优化的6N维目标函数
    :param B: 重建后灰度图数组
    :param Di: 原始深度图数组
    :param I: 原始灰度图数组
    :param D: 增强中的深度图数组
    :param w: 超参数
    :return: 改变后的深度图数组
    '''
    wg = int(w[0])
    ws = int(w[1])
    wp = int(w[2])

    # 对目标函数F进行更新
    for i in range(0, N):
        #排除掉第一行最后一行第一列最后一列
        if i % Nx == 0 or i % Nx == Nx-1:
            continue
        if i >= 0 and i < Nx:
            continue
        if i >= N-Nx and i < N:
            continue
        F[0][i] = constrains.ShadingGradients1(B, I, i, wg)
    for i in range(N, 2*N):
        flag = i - N
        # 排除掉第一行最后一行第一列最后一列
        if flag % Nx == 0 or flag % Nx == Nx - 1:
            continue
        if flag >= 0 and flag < Nx:
            continue
        if flag >= N - Nx and flag < N:
            continue
        F[0][i] = constrains.ShadingGradients2(B, I, i-N, wg)
    for i in range(2*N, 3*N):
        flag = i - 2*N
        # 排除掉第一行最后一行第一列最后一列
        if flag % Nx == 0 or flag % Nx == Nx - 1:
            continue
        if flag >= 0 and flag < Nx:
            continue
        if flag >= N - Nx and flag < N:
            continue
        F[0][i] = constrains.SmoothConstraint1(D, i-2*N, ws)
    for i in range(3*N, 4*N):
        flag = i - 3 * N
        # 排除掉第一行最后一行第一列最后一列
        if flag % Nx == 0 or flag % Nx == Nx - 1:
            continue
        if flag >= 0 and flag < Nx:
            continue
        if flag >= N - Nx and flag < N:
            continue
        F[0][i] = constrains.SmoothConstraint2(D, i-3*N, ws)
    for i in range(4*N, 5*N):
        flag = i - 4 * N
        # 排除掉第一行最后一行第一列最后一列
        if flag % Nx == 0 or flag % Nx == Nx - 1:
            continue
        if flag >= 0 and flag < Nx:
            continue
        if flag >= N - Nx and flag < N:
            continue
        F[0][i] = constrains.SmoothConstraint3(D, i-4*N, ws)
    for i in range(5*N, 6*N):
        flag = i - 5 * N
        # 排除掉第一行最后一行第一列最后一列
        if flag % Nx == 0 or flag % Nx == Nx - 1:
            continue
        if flag >= 0 and flag < Nx:
            continue
        if flag >= N - Nx and flag < N:
            continue
        F[0][i] = constrains.DepthConstraint(D, Di, i-5*N, wp)

    print("F finished!")
    np.savetxt(fname="F.csv",X=F,fmt="%f",delimiter=",")
    J = Jacobi(D, w, l)
    print("J finished!")
    print(np.matmul(J.T, J))
    mid = np.matmul(J.T, J)
    print(mid.shape)
    inv = InvLU.InvFunc(mid)
    print("inv finished")
    d = -np.matmul(np.matmul(inv, J.T), F)
    D = D + d
    print("D finished!")
    return D

