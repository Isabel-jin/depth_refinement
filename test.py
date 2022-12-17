import cv2 as cv
from PIL import Image
import numpy as np
import LMS2

N = 154 * 127

sampler = 3

l = [51.040569,
     -11.897069,
     -78.228003,
     -18.870133,
     0.716704,
     -34.047397,
     -9.706077,
     6.701684,
     0.406902]

#获取并套用mask
def masker(D, img):
    mask = np.zeros((D.shape[0], D.shape[1]))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i][j] != 0 and D[i-1][j] != 0 and D[i][j-1] != 0:
                mask[i][j] = 1
    img = img * mask
    return img

if __name__ == '__main__':
    B_img = cv.imread("./processed/cut/B.png", cv.IMREAD_GRAYSCALE)
    D_img = np.array(Image.open('./processed/cut/D_cut.png'))
    I_img = cv.imread("./processed/cut/I.png", cv.IMREAD_GRAYSCALE)

    # 为防止雅可比矩阵内存超限，暂时先对图像下采样
    height = B_img.shape[0]
    width = B_img.shape[1]

    B_new = np.zeros((height//sampler + 1, width//sampler + 1), np.uint8)
    D_new = np.zeros((height//sampler + 1, width//sampler + 1), np.int32)
    I_new = np.zeros((height//sampler + 1, width//sampler + 1), np.uint8)
    for i in range(height):
        for j in range(width):
            if i % sampler == 0 and j % sampler == 0:
                B_new[i//sampler, j//sampler] = B_img[i, j]
                D_new[i // sampler, j // sampler] = D_img[i, j]
                I_new[i // sampler, j // sampler] = I_img[i, j]

    cv.imshow('B', B_new)
    cv.imshow('I', I_new)
    print(B_new.shape)
    cv.waitKey(0)


    # 图像一维化
    B = B_img.ravel()   # 重建后灰度数组
    Di = D_img.ravel()  # 原始深度数组
    I = I_img.ravel()   # 原始灰度数组
    D = Di.copy()  # 用原始深度数组初始化待优化的深度数组
    F = np.zeros((1, 6*N))   # 6N维待优化目标函数

    tester = 1
    res = np.zeros((tester, N))
    # tester次优化
    for i in range(tester):
        LMS2.GaussNewton(F, B, Di, I, D, (1, 400, 10))
        for j in range(N):
            res[i][j] = D[j] - Di[j]
            cv.imshow('res', res[i])
            cv.waitKey(0)
