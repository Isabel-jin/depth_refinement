import math
import numpy as np
#计算球谐函数的代码

#一次阶乘函数
def fractorial(v):
    if v == 0:
        return 1
    result = v
    v = v - 1
    while v > 0:
        result *= v
        v = v - 1
    return result


#二次阶乘函数
def DoubleFractorial(x):
    if x == 0 or x == -1:
        return 1
    result = x
    x = x - 2
    while x > 0:
        result *= x
        x = x - 2
    return result


#归一化系数Klm计算
def K(l, m):
    return math.sqrt(((2 * l + 1) * fractorial(l - m)) / (4 * math.pi * fractorial(l + m)))

#递归计算x(cos(theta))的连带勒让德多项式
def P(l, m, x):
    if l == m:
        return math.pow(-1.0, m) * DoubleFractorial(2 * m - 1) * math.pow(math.sqrt(1 - x * x), m)
    if l == m + 1:
        return x * (2 * m + 1) * P(m, m, x)
    return (x * (2 * l - 1) * P(l - 1, m, x) - (l + m - 1) * P(l - 2, m, x)) / (l - m)

#l,m阶球谐函数Y
def Ylm(l, m, theta, phi):
    if m == 0:
        return K(l, 0) * P(l, 0, math.cos(theta))
    if m > 0:
        return math.sqrt(2.0) * K(l, m) * math.cos(m * phi) * P(l, m, math.cos(theta))
    #如果m小于0，预先乘上-1
    return math.sqrt(2.0) * K(l, -m) * math.cos(-m * phi) * P(l, -m, math.cos(theta))


# 对特定的阶数的一组正交基
def Basis(theta, phi):
    degree = 2
    n = 9
    Y = np.zeros(n)
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            Y[l * (l + 1) + m] = Ylm(l, m, theta, phi)
    return Y


# 用球谐函数重构theta phi方向的环境光
def Render(theta, phi, lk):
    #二阶九项
    degree = 2
    n = 9
    Y = Basis(theta, phi)
    color = 0
    for i in range(n):
        color = color + Y[i] * lk[i]
        if color > 100:
            print("error")
    return color

