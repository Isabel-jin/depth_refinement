import numpy as np

# 下采样
sampler = 5

# 相机内参
fx = 975.9293/sampler
fy = 975.9497/sampler
ux = 380 / sampler / 2
uy = 460 / sampler / 2


def Partial_loss2E(N, w, mask):
    '''
    :param N: 总点数
    :param w: 超参数
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :return:
        dloss_dE: 1x6N的矩阵，loss对E的偏导
    '''
    dloss_dE = np.zeros(6 * N)

    for n in range(N):
        if mask[n] == 0:
            continue
        dloss_dE[n] = w[0]
        dloss_dE[n + N] = w[0]
        dloss_dE[n + 2 * N] = w[1]
        dloss_dE[n + 3 * N] = w[1]
        dloss_dE[n + 4 * N] = w[1]
        dloss_dE[n + 5 * N] = w[2]

    return dloss_dE


def Partial_E2B(I, B, mask, Nk):
    '''
    :param I: 1xN的矩阵，原始灰度图
    :param B: 1xN的矩阵，重构后的灰度图
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :param Nk: 列数
    :return:
        dE_dB: 6NxN的矩阵，E对 B的偏导数
    '''
    N = I.shape[0]
    dE_dB = np.zeros((6*N, N))
    for n in range(N):
        if mask[n] == 0:
            continue
        # 阴影梯度约束第一项
        dE_dB[n][n] = 2 * (int(B[n]) - int(B[n + 1]) - (int(I[n]) - int(I[n + 1])))
        dE_dB[n][n+1] = - 2 * (int(B[n]) - int(B[n + 1]) - (int(I[n]) - int(I[n + 1])))
        # 阴影梯度约束第二项
        dE_dB[n + N][n] = 2 * (int(B[n]) - int(B[n + Nk]) - (int(I[n]) - int(I[n + Nk])))
        dE_dB[n + N][n + Nk] = - 2 * (int(B[n]) - int(B[n + Nk]) - (int(I[n]) - int(I[n + Nk])))

    return dE_dB


def Partial_E2p3(Nx, p3, mask):
    '''
    :param Nx: 列数
    :param p3: 3xN的矩阵，三维点云
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :return:
        dE_dp3: 6Nx3N的矩阵，E对 p3的偏导数
    '''
    N = p3.shape[1]
    # 将p3转成1x3N矩阵
    p3 = p3.ravel()
    dE_dp3 = np.zeros((6 * N, 3 * N))
    # 平滑约束第num项
    for num in range(3):
        # 对该约束项两个矩阵的起始项
        pin = N * num
        Ein = (2+num) * N
        for n in range(N):
            if mask[n] == 0:
                continue
            pn = pin + n
            En = Ein + n
            dE_dp3[En][pn] = 2 * (p3[pn] - 0.25 * (p3[pn - 1] + p3[pn + 1] + p3[pn - Nx] + p3[pn + Nx]))
            dE_dp3[En][pn-1] = 2 * (-0.25) * (p3[pn] - 0.25 * (p3[pn - 1] + p3[pn + 1] + p3[pn - Nx] + p3[pn + Nx]))
            dE_dp3[En][pn+1] = 2 * (-0.25) * (p3[pn] - 0.25 * (p3[pn - 1] + p3[pn + 1] + p3[pn - Nx] + p3[pn + Nx]))
            dE_dp3[En][pn-Nx] = 2 * (-0.25) * (p3[pn] - 0.25 * (p3[pn - 1] + p3[pn + 1] + p3[pn - Nx] + p3[pn + Nx]))
            dE_dp3[En][pn+Nx] = 2 * (-0.25) * (p3[pn] - 0.25 * (p3[pn - 1] + p3[pn + 1] + p3[pn - Nx] + p3[pn + Nx]))

    return dE_dp3


def PartialE2D(D, Di, mask):
    '''
    :param D: 1xN的矩阵，待优化的深度图
    :param Di: 1xN的矩阵，原始深度图
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :return:
        dE_dD: 6NxN的矩阵，E对 D的直接偏导数
    '''
    N = D.shape[0]
    dE_dD = np.zeros((6 * N, N))
    for n in range(N):
        if mask[n] == 0:
            continue
        dE_dD[n + 5 * N][n] = 2 * (D[n] - Di[n])
    return dE_dD


def Partial_B2H(B, mask, l, albedo):
    '''
    :param B: 1xN的矩阵，重新生成的灰度图
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :param l: 1x9的矩阵，球谐光照系数
    :param albedo: 反照率
    :return:
        dB_dH: Nx9N的矩阵，B对 H的偏导
    '''
    N = B.shape[0]
    dB_dH = np.zeros((N, 9 * N))
    for k in range(9):
        for n in range(N):
            if mask[n] == 0:
                continue
            kn = k * N + n
            dB_dH[n][kn] = albedo * l[k]
    return dB_dH


def Partial_H2normal(mask, normal):
    '''
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :param normal: 3xN的矩阵，各个方向的法向量
    :return:
        dH_dnormal: 9Nx3N的矩阵，H对 normal的偏导
    '''
    N = normal.shape[1]
    dH_dnormal = np.zeros((9 * N, 3 * N))
    # n点的 H 只受到 n 点的 normal 影响
    for n in range(N):
        if mask[n] == 0:
            continue
        # H0项：前n行均为0
        # H1项
        dH_dnormal[N + n][N + n] = 1
        # H2项
        dH_dnormal[2 * N + n][2 * N + n] = 1
        # H3项
        dH_dnormal[3 * N + n][n] = 1
        # H4项
        dH_dnormal[4 * N + n][n] = normal[1][n]
        dH_dnormal[4 * N + n][N + n] = normal[0][n]
        # H5项
        dH_dnormal[5 * N + n][2 * N + n] = normal[1][n]
        dH_dnormal[5 * N + n][N + n] = normal[2][n]
        # H6项
        dH_dnormal[6 * N + n][n] = -2 * normal[0][n]
        dH_dnormal[6 * N + n][N + n] = -2 * normal[1][n]
        dH_dnormal[6 * N + n][2 * N + n] = 4 * normal[2][n]
        # H7项
        dH_dnormal[7 * N + n][2 * N + n] = normal[0][n]
        dH_dnormal[7 * N + n][n] = normal[2][n]
        # H8项
        dH_dnormal[8 * N + n][n] = 2 * normal[0][n]
        dH_dnormal[8 * N + n][N + n] = -2 * normal[1][n]

    return dH_dnormal


def Partial_normal2D(mask, Nx, D):
    '''
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :param Nx: 每行列数
    :param D: 1xN的矩阵，深度图 D
    :return:
        dnormal_dD: 3NxN的矩阵，法向量对深度图的偏导
    '''
    N = D.shape[0]
    dnormal_dD = np.zeros((3 * N, N))
    for n in range(N):
        if mask[n] == 0:
            continue
        # x方向
        dnormal_dD[n][n] = D[n - Nx] / fy  # dnx/dD(i,j)
        dnormal_dD[n][n - Nx] = (D[n] - D[n - 1]) / fy  # dnx/dD(i,j-1)
        dnormal_dD[n][n - 1] = -D[n - Nx] / fy  # dnx/dD(i-1,j)

        # y方向
        dnormal_dD[n + N][n] = D[n - 1] / fx
        dnormal_dD[n + N][n - Nx] = -D[n - 1] / fx
        dnormal_dD[n + N][n - 1] = (D[n] - D[n - Nx]) / fx

        # z方向
        j = n // Nx
        i = n - Nx * j
        a = (ux - i) / fx
        b = (uy - j) / fy
        dnormal_dD[n + 2 * N][n] = a * dnormal_dD[n][n] + b * dnormal_dD[n + N][n]
        dnormal_dD[n + 2 * N][n - Nx] = a * dnormal_dD[n][n - Nx] + b * dnormal_dD[n + N][n - Nx] - D[n - 1] / (fx * fy)
        dnormal_dD[n + 2 * N][n - 1] = a * dnormal_dD[n][n - 1] + b * dnormal_dD[n + N][n - 1] - D[n - Nx] / (fx * fy)

    return dnormal_dD


def Partial_p32D(mask, D, Nx):
    '''
    :param mask: 1xN的矩阵，正向传播时生成的 mask
    :param D: 1xN的矩阵，重新生成的深度图
    :param Nx: 每行列数
    :return:
        dp3_dD: 3NxN的矩阵，三维点云对深度图的求导
    '''
    N = D.shape[0]
    dp3_dD = np.zeros((3 * N, N))
    for n in range(N):
        if mask[n] == 0:
            continue
        j = n // Nx
        i = n - Nx * j
        a = (ux - i) / fx
        b = (uy - j) / fy
        dp3_dD[n][n] = a
        dp3_dD[n + N][n] = b
        dp3_dD[n + 2 * N][n] = 1

    return dp3_dD






