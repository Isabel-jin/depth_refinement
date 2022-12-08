import numpy as np

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


def ShadingGradients1(B, I, n, wg):
    '''
    阴影梯度约束第一项
    :param B: 重建后的灰度图
    :param I: 原始灰度图
    :param n: 索引
    :param wg: 超参数
    :return: 阴影梯度约束第一项
    '''
    return pow(wg, 0.5) * (B[n] - B[n + Nx] - (I[n] - I[n + Nx]))


def ShadingGradients2(B, I, n, wg):
    '''
    阴影梯度约束第二项
    :param B: 重建后的灰度图
    :param I: 原始灰度图
    :param n: 索引
    :param wg: 超参数
    :return: 阴影梯度约束第二项
    '''
    return pow(wg, 0.5) * (B[n] - B[n + 1] - (I[n] - I[n + 1]))


def SmoothConstraint1(D, n, ws):
    '''
    平滑约束第一项
    :param B: 重建后的灰度图
    :param n: 索引
    :param ws: 超参数
    :return: 平滑约束第一项
    '''
    i = n // Nx
    r3 = D[n] - ws * (D[n - Nx] + D[N - 1] + D[n + Nx] + D[n + 1])
    r1 = (i - ux) / fx * r3
    return pow(ws, 0.5) * r1


def SmoothConstraint2(D, n, ws):
    '''
    平滑约束第二项
    :param B: 重建后的灰度图
    :param n: 索引
    :param ws: 超参数
    :return: 平滑约束第二项
    '''
    i = n // Nx
    j = n - i * Nx
    r3 = D[n] - ws * (D[n - Nx] + D[n - 1] + D[n + Nx] + D[n + 1])
    r2 = (j - uy) / fy * r3
    return pow(ws, 0.5) * r2


def SmoothConstraint3(D, n, ws):
    '''
    平滑约束第三项
    :param B: 重建后的灰度图
    :param n: 索引
    :param ws: 超参数
    :return: 平滑约束第三项
    '''
    r3 = D[n] - ws * (D[n - Nx] + D[n - 1] + D[n + Nx] + D[n + 1])
    return pow(ws, 0.5) * r3


def DepthConstraint(D, Di, n, wp):
    '''
    :param D: 深度数组
    :param Di: 原始灰度图
    :param n: 索引
    :param wp: 超参数
    :return:
    '''
    return pow(wp, 0.5) * (D[n] - Di[n])
