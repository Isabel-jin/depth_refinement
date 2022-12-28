import numpy as np
import math

# 下采样
sampler = 5

# 相机内参
fx = 975.9293/sampler
fy = 975.9497/sampler
ux = 380 / sampler / 2
uy = 460 / sampler / 2

# 考虑所有的点都套用统一的mask

def D2n(D, Nx):
    '''
    :param D: 1xN的矩阵，待优化的深度
    :param Nx: 列数（每行点数）
    :return:
        normal: 3xN的矩阵，法向量
        mask_n: 1xN的矩阵，计算法向量过程中的mask
    '''
    N = D.shape[0]
    mask_n = np.ones(N)
    normal = np.zeros((3, N))

    for n in range(N):
        # i代表x方向坐标，j代表y方向坐标
        j = n // Nx
        i = n - Nx * j
        # 计算法向量时排除第一行和第一列
        if i == 0 or j == 0:
            mask_n[n] = 0
            continue

        # 分别求第n个点的x,y,z方向法向量
        normal[0][n] = D[n-Nx]*(D[n]-D[n-1])/fy
        normal[1][n] = D[n-1]*(D[n]-D[n-Nx])/fx

        a = (ux-i)/fx
        b = (uy-j)/fy
        normal[2][n] = a*normal[0][n] + b*normal[1][n] - D[n-Nx]*D[n-1]/(fx*fy)

        # 归一化
        m = math.sqrt(pow(normal[0][n], 2) + pow(normal[1][n], 2) + pow(normal[2][n], 2))
        if m != 0:
            normal[0][n] /= m
            normal[1][n] /= m
            normal[2][n] /= m

    return normal, mask_n


def n2H(normal, mask_n):
    '''
    :param normal: 3xN的矩阵，法向量
    :param mask_n: 1xN的矩阵，计算法向量过程中的mask
    :return:
        H: 9xN的矩阵，法向量球谐系数
    '''
    N = normal.shape[1]
    H = np.zeros((9, N))
    for n in range(N):
        # 排除掉没办法计算法向量的点
        if mask_n[n] == 0:
            continue

        H[0][n] = 1
        H[1][n] = normal[1][n]
        H[2][n] = normal[2][n]
        H[3][n] = normal[0][n]
        H[4][n] = normal[0][n] * normal[1][n]
        H[5][n] = normal[2][n] * normal[1][n]
        H[6][n] = -pow(normal[0][n], 2) - pow(normal[1][n], 2) + 2 * pow(normal[2][n], 2)
        H[7][n] = normal[2][n] * normal[0][n]
        H[8][n] = pow(normal[0][n], 2) - pow(normal[1][n], 2)

    return H


def H2B(H, l, albedo, mask_n):
    '''
    :param H: 9xN的矩阵，法向量球谐系数
    :param l: 9个元素的向量，光照系数
    :param albedo: 反照率
    :param mask_n: 1xN的矩阵，计算法向量过程中的mask
    :return:
        B: 1xN的矩阵，重构的灰度图像
    '''
    N = H.shape[1]
    B = np.zeros(N)

    for n in range(N):
        # 排除掉没办法计算法向量的点
        if mask_n[n] == 0:
            continue
        sum = 0
        for k in range(9):
            sum = sum + l[k] * H[k][n]
        B[n] = albedo * sum

    return B


def D2p3(D, Nx):
    '''
    :param D: 1xN的矩阵，待优化的深度
    :param Nx: 列数（每行点数）
    :return:
        p3: 3xN的矩阵，三维点云坐标
    '''
    N = D.shape[0]
    p3 = np.zeros((3, N))

    for n in range(N):
        # i代表x方向坐标，j代表y方向坐标
        j = n // Nx
        i = n - Nx * j
        # 分别求第n个点的x,y,z三维点云坐标
        p3[0][n] = D[n] * (i - ux)/fx
        p3[1][n] = D[n] * (j - uy)/fy
        p3[2][n] = D[n]
    return p3


def B2E1(B, I, mask_n, Nx):
    '''
    阴影梯度约束第一项
    :param B: 1xN的矩阵，重建后的灰度图
    :param I: 1xN的矩阵，原始灰度图
    :param mask_n: 1xN的矩阵，计算法向量时的 mask
    :param Nx: 列数
    :return:
        E1: 1xN的矩阵，阴影梯度约束第一项
        mask_E1: 1xN的矩阵，计算E1时的 mask
    '''
    N = B.shape[0]
    E1 = np.zeros(N)
    mask_E1 = np.ones(N)
    for n in range(N):
        # i代表x方向坐标，j代表y方向坐标
        j = n // Nx
        i = n - Nx * j
        if mask_n[n] == 0:
            mask_E1[n] = 0
            continue
        # 计算E1时排除掉最后一列
        if i == Nx - 1:
            mask_E1[n] = 0
            continue
        E1[n] = pow((int(B[n]) - int(B[n + 1]) - (int(I[n]) - int(I[n + 1]))), 2)
    return E1, mask_E1


def B2E2(B, I, mask_n, Nx):
    '''
    阴影梯度约束第二项
    :param B: 1xN的矩阵，重建后的灰度图
    :param I: 1xN的矩阵，原始灰度图
    :param mask_n: 1xN的矩阵，计算法向量时的 mask
    :param Nx: 列数
    :return:
        E2: 1xN的矩阵，阴影梯度约束第一项
        mask_E2: 1xN的矩阵，计算E1时的 mask
    '''
    N = B.shape[0]
    E2 = np.zeros(N)
    mask_E2 = np.ones(N)
    for n in range(N):
        # i代表x方向坐标，j代表y方向坐标
        j = n // Nx
        i = n - Nx * j
        if mask_n[n] == 0:
            mask_E2[n] = 0
            continue
        # 计算E2时排除掉最后一行
        if j == N//Nx - 1:
            mask_E2[n] = 0
            continue
        E2[n] = pow((int(B[n]) - int(B[n + Nx]) - (int(I[n]) - int(I[n + Nx]))), 2)
    return E2, mask_E2


def p32E345(p3, Nx):
    '''
    三项平滑约束
    :param p3: 3xN的矩阵，三维点云
    :param Nx: 列数
    :return:
        E1: 1xN的矩阵，阴影梯度约束第一项
        mask_E1: 1xN的矩阵，计算E1时的 mask
    '''
    N = p3.shape[1]
    E345 = np.zeros(3 * N)
    mask_E345 = np.ones(N)
    for n in range(N):
        # i代表x方向坐标，j代表y方向坐标
        j = n // Nx
        i = n - Nx * j
        # 计算E345时排除掉首行首列尾行尾列
        if i == Nx - 1 or i == 0 or j == N//Nx - 1 or j == 0:
            mask_E345[n] = 0
            continue
        E345[n] = pow((p3[0][n] - 0.25 * (p3[0][n-1] + p3[0][n+1] + p3[0][n-Nx] + p3[0][n+Nx])), 2)
        E345[n+N] = pow((p3[1][n] - 0.25 * (p3[1][n - 1] + p3[1][n + 1] + p3[1][n - Nx] + p3[1][n + Nx])), 2)
        E345[n+2*N] = pow((p3[2][n] - 0.25 * (p3[2][n - 1] + p3[2][n + 1] + p3[2][n - Nx] + p3[2][n + Nx])), 2)
    return E345, mask_E345


def D2E6(D, Di, Nx):
    '''
    深度约束
    :param D: 1xN的矩阵，优化后的深度图
    :param Di: 1xN的矩阵，原始深度图
    :param Nx: 列数
    :return:
        E6: 1xN的矩阵，深度约束项
    '''
    N = D.shape[0]
    E6 = np.zeros(N)
    for n in range(N):
        E6[n] = pow((D[n] - Di[n]), 2)
    return E6


def E2loss(E1, E2, E345, E6, w):
    '''
    :param E1: 1xN的矩阵，阴影梯度约束第一项
    :param E2: 1xN的矩阵，阴影梯度约束第二项
    :param E345: 1x3N的矩阵，平滑约束
    :param E6: 1xN的矩阵，深度约束
    :param w: 3个元素的向量，超参数
    :return:
        loss: 损失函数
        E: 1x6N的矩阵，总约束项
    '''
    # mask覆盖的值本来就被置为0，对loss不产生影响
    loss = w[0] * np.sum(E1) + w[0] * np.sum(E2) + w[1] * np.sum(E345) + w[2] * np.sum(E6)
    E = np.concatenate((E1, E2, E345, E6))
    return E, loss


