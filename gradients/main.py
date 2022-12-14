import cv2 as cv
from PIL import Image
import numpy as np
import forwarding as fwd
import BP
import time
import matplotlib.pyplot as plt

# 每次修改降采样sampler要三个文件都改

# 下采样
sampler = 5

l = [51.040569,
     -11.897069,
     -78.228003,
     -18.870133,
     0.716704,
     -34.047397,
     -9.706077,
     6.701684,
     0.406902]


# 法向量图生成与存储
def SaveNormal(normal, Nx, timer):
    # 显示法向量图
    normal1 = normal[0, :].reshape(-1, Nx)
    normal2 = normal[1, :].reshape(-1, Nx)
    normal3 = normal[2, :].reshape(-1, Nx)
    normaler = np.dstack((normal1, normal2, normal3))
    normaler -= np.min(normaler)
    normaler = (normaler / np.max(normaler) * 255).astype(np.uint8)
    cv.imwrite("./results/normal/normal"+str(timer)+".jpg", normaler)
'''
    cv.imwrite("./results/normal1.jpg", normaler[:, :, 0])
    cv.imwrite("./results/normal2.jpg", normaler[:, :, 1])
    cv.imwrite("./results/normal3.jpg", normaler[:, :, 2])
'''


# 可视化三维点云
def VisualPoints(points, Nx):
    x = points[0, :].reshape(-1, Nx)  # x position of point
    y = points[1, :].reshape(-1, Nx)  # y position of point
    z = points[2, :].reshape(-1, Nx)  # z position of point
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,   # x
               y,   # y
               z,   # z
               c=z, # height data for color
               cmap='rainbow',
               marker="x")
    ax.axis()
    plt.show()


if __name__ == '__main__':
    # 清空文件内容
    f = open('results.txt', 'w')
    f.truncate()
    f.close()
    D_img = np.array(Image.open('../processed/cut/D_cut.png'))
    I_img = cv.imread("../processed/cut/I.png", cv.IMREAD_GRAYSCALE)

    # 为防止雅可比矩阵内存超限，暂时先对图像下采样
    height = I_img.shape[0]
    width = I_img.shape[1]

    # 针对能否整除两种情况生成合适的图像大小
    flag1 = int(not(height % sampler == 0))
    flag2 = int(not(width % sampler == 0))
    D_new = np.zeros((height//sampler + flag1, width//sampler + flag2), np.int32)
    I_new = np.zeros((height//sampler + flag1, width//sampler + flag2), np.uint8)
    for i in range(height):
        for j in range(width):
            if i % sampler == 0 and j % sampler == 0:
                D_new[i // sampler, j // sampler] = D_img[i, j]
                I_new[i // sampler, j // sampler] = I_img[i, j]

    Nx = I_new.shape[1]
    N = I_new.shape[0] * I_new.shape[1]
    print(I_new.shape)
    # 图像一维化
    Di = D_new.ravel()  # 原始深度数组
    I = I_new.ravel()   # 原始灰度数组
    cv.imwrite("./results/I.jpg", I.reshape((-1, Nx)))
    D = np.array(Di.copy())  # 用原始深度数组初始化待优化的深度数组
    F = np.zeros((1, 6*N))   # 6N维待优化目标函数

    w = [1, 400, 1000]
    albedo = 1
    mask = np.ones(N)
    timers = 250
    # step 设定每一轮的步长
    step = 0.0000001 * np.ones(timers)
    loss = np.zeros(timers)
    loss1 = np.zeros(timers)
    loss2 = np.zeros(timers)
    loss3 = np.zeros(timers)
    # 梯度下降循环
    for timer in range(timers):
        f = open('results.txt', 'a')
        start = time.process_time()
        # 正向传播，刷新各个参量
        normal, mask_n = fwd.D2n(D, Nx)             # 得到3xN的矩阵normal
        # 显示法向量图
        SaveNormal(normal, Nx, timer)

        H = fwd.n2H(normal, mask_n)                 # 得到9xN的矩阵H
        B = fwd.H2B(H, l, albedo, mask_n)           # 得到1xN的矩阵B，优化后的灰度图
        cv.imwrite("./results/B/B"+str(timer)+".jpg", B.reshape((-1, Nx)).astype(np.uint8))
        res = np.absolute(B - I)
        cv.imwrite("./results/res/res"+str(timer)+".jpg", res.reshape((-1, Nx)).astype(np.uint8))
        p3 = fwd.D2p3(D, Nx)                        # 得到3xN的矩阵p3
        # 可视化三维点云
        # VisualPoints(p3, Nx)
        E1, mask_E1 = fwd.B2E1(B, I, mask_n, Nx)
        E2, mask_E2 = fwd.B2E2(B, I, mask_n, Nx)
        E345, mask_E345 = fwd.p32E345(p3, Nx)
        E6 = fwd.D2E6(D, Di, Nx)
        E, loss[timer] = fwd.E2loss(E1, E2, E345, E6, w)   # 得到6xN的矩阵E以及数值loss
        mask = cv.bitwise_and(cv.bitwise_and(mask_E1, mask_E2), mask_E345)
        # 与运算得到总的mask为1xN的矩阵

        loss1[timer] = str(w[0] * (np.sum(E1) + np.sum(E2)))
        loss2[timer] = str(w[1] * np.sum(E345))
        loss3[timer] = str(w[2] * np.sum(E6))

        print("loss1: "+str(loss1[timer]))
        print("loss2: "+str(loss2[timer]))
        print("loss3: "+str(loss3[timer]))
        print("loss: "+str(loss[timer]))

        f.write("timer: "+str(timer)+' ')
        f.write("loss1: "+str("%12.5f" % loss1[timer])+' ')
        f.write("loss2: "+str("%12.5f" % loss2[timer])+' ')
        f.write("loss3: "+str("%12.5f" % loss3[timer])+' ')
        f.write("loss: "+str("%12.5f" % loss[timer])+' ')

        # 反向传播，求解梯度
        dloss_dE = BP.Partial_loss2E(N, w, mask)    # 1x6N的矩阵
        dE_dB = BP.Partial_E2B(I, B, mask, Nx)      # 6NxN的矩阵
        dE_dp = BP.Partial_E2p3(Nx, p3, mask)       # 6Nx3N的矩阵
        dE_dD = BP.PartialE2D(D, Di, mask)          # 6NxN的矩阵
        dB_dH = BP.Partial_B2H(B, mask, l, albedo)  # Nx9N的矩阵
        dH_dn = BP.Partial_H2normal(mask, normal)   # 9Nx3N的矩阵
        dn_dD = BP.Partial_normal2D(mask, Nx, D)    # 3NxN的矩阵
        dp_dD = BP.Partial_p32D(mask, D, Nx)        # 3NxN的矩阵
        print("yes")

        dloss_dD = np.dot(np.dot(np.dot(np.dot(dloss_dE, dE_dB), dB_dH), dH_dn), dn_dD) + \
                                     np.dot(np.dot(dloss_dE, dE_dp), dp_dD) + np.dot(dloss_dE, dE_dD)

        D = D - dloss_dD * step[timer]

        end = time.process_time()
        print("run time: "+str(end - start))
        f.write("run time: "+str("%3.2f" % (end - start))+'\n')
        f.close()

    # 绘制loss曲线
    x = range(timers)
    plt.plot(x, loss, label='loss')
    plt.plot(x, loss1, label='loss1', color='r')
    plt.plot(x, loss2, label='loss2', color='g')
    plt.plot(x, loss3, label='loss3', color='b')
    plt.xlabel('timers')
    plt.ylabel('loss')
    plt.savefig('loss.png')
    plt.close()

    plt.plot(x, loss, label='loss')
    plt.savefig('loss_show.png')
    plt.close()








