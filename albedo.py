import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def hist(img):
    # 显示图像所有颜色直方图，各个颜色直方图
    plt.figure(figsize=(10, 8))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.subplot(2, 2, 1)
    plt.hist(gray.ravel(), 256, [0, 255])  # img.ravel将多维数组变成一维数组
    plt.title('gray')

    plt.subplot(2, 2, 2)
    plt.hist(img[:, :, 0].ravel(), 256, [0, 255], color='blue')  # img.ravel将多维数组变成一维数组
    plt.title('blue')

    plt.subplot(2, 2, 3)
    plt.hist(img[:, :, 1].ravel(), 256, [0, 255], color='green')  # img.ravel将多维数组变成一维数组
    plt.title('green')

    plt.subplot(2, 2, 4)
    plt.hist(img[:, :, 2].ravel(), 256, [0, 255], color='red')  # img.ravel将多维数组变成一维数组
    plt.title('red')
    plt.savefig(r'./result/albedo/new/cut/albedo_hist.jpg')


def grad(img):
    # 利用sobel核进行两个方向的梯度滤波
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    cv.imwrite('./result/albedo/new/cut/sobel_x.jpg', sobelx)
    cv.imwrite('./result/albedo/new/cut/sobel_y.jpg', sobely)

    plt.figure()
    plt.subplot(211)
    plt.hist(sobelx.ravel(), 256, [0, 255])
    plt.title('sobelx_hist')
    plt.subplot(212)
    plt.hist(sobely.ravel(), 256, [0, 255])
    plt.title('sobely_hist')

    plt.savefig(r'./result/albedo/new/cut/sobel_hist.jpg')
    return sobelx, sobely

if __name__ == '__main__' :
    img = cv.imread('./processed/cut/Ic.jpg', cv.IMREAD_COLOR)
    gray = cv.imread('./processed/cut/B.png', cv.IMREAD_GRAYSCALE)
    cv.imwrite('./result/albedo/new/cut/gray.jpg', gray)
    print(img.shape)
    print(gray.shape)

    length = gray.shape[0]
    width = gray.shape[1]
    albedo = np.zeros((length, width, 3), np.float32)

    # 排除灰度图中的零点，防止做除法报错
    for j in range(length):
        for k in range(width):
            if gray[j, k] == 0:
                gray[j, k] = 1

    # 本质图像分解计算albedo
    for i in range(3):
        albedo[:, :, i] = img[:, :, i] / gray * 150  # 150是实验得到的参数

    cv.imwrite('./result/albedo/new/cut/albedo.jpg', albedo)
    # albedo三个通道的直方图
    hist(albedo)

    albedo = albedo.astype(np.uint8)
    albedo_gray = cv.cvtColor(albedo, cv.COLOR_BGR2GRAY)
    cv.imwrite('./result/albedo/new/cut/albedo_gray.jpg', albedo_gray)

    # 求xy两个方向的梯度
    sobelx, sobely = grad(albedo_gray)

    # 设定特定的阈值进行二值化（50为参数）
    ret, binary_x = cv.threshold(sobelx, 0, 255, cv.THRESH_BINARY_INV)
    cv.imwrite('./result/albedo/new/cut/binary_x.jpg', binary_x)

    # 设定特定的阈值
    ret, binary_y = cv.threshold(sobely, 0, 255, cv.THRESH_BINARY_INV)
    cv.imwrite('./result/albedo/new/cut/binary_y.jpg', binary_y)
