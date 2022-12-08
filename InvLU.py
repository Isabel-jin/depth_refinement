import numpy as np
import copy

# 定义类 功能均在类内实现
class PLU():
    def __init__(self, Mat):
        self.Mat = copy.deepcopy(Mat)  # 深层拷贝，防止初始矩阵内存被重复占用，下同
        self.Mat_stable = copy.deepcopy(Mat)
        self.Mat_R = Mat.shape[0]
        self.Mat_C = Mat.shape[1]
        self.Mat_P = np.array([i for i in range(self.Mat_R)])
        self.Mat_P_result = np.zeros([Mat.shape[0], Mat.shape[1]], dtype=np.float32)
        self.Mat_L = np.eye(Mat.shape[0], dtype=np.float32)
        self.Mat_U = np.zeros([Mat.shape[0], Mat.shape[1]], dtype=np.float32)
        self.Mat_inversion = np.zeros([Mat.shape[0], Mat.shape[1]], dtype=np.float32)

    # 更新矩阵，始终将该次计算的最大主元移到最上方
    def update_mat(self, row):
        Mat_this = abs(self.Mat)
        max_num = np.max(Mat_this[row:, row])
        max_index = np.where(Mat_this[row:, row] == max_num)[0] + row
        if row == max_index:
            return self.Mat
        else:
            change = self.Mat[max_index, :]
            self.Mat[max_index, :] = self.Mat[row, :]
            self.Mat[row, :] = change
            change_P = self.Mat_P[max_index]
            self.Mat_P[max_index] = self.Mat_P[row]
            self.Mat_P[row] = change_P
        return self.Mat

    # 分解矩阵，当detail=True时输出分解过程
    def Factorization(self, detail=True):
        for i in range(self.Mat_R - 1):
            self.Mat = self.update_mat(i)
            if detail:
                print("第{}次更新".format(i), "\n", self.Mat, "\n", self.Mat_P)
            for j in range(self.Mat_R - i - 1):
                if self.Mat[i + j + 1, i] == 0:
                    self.Mat = self.Mat
                else:
                    ratio = self.Mat[i + j + 1, i] / self.Mat[i, i]
                    self.Mat[i + j + 1, i + 1:] = self.Mat[i + j + 1, i + 1:] - ratio * self.Mat[i, i + 1:]
                    self.Mat[i + j + 1, i] = ratio
            if detail:
                print("第{}次计算".format(i), "\n", self.Mat)

    # 输出分解后的P,L,U矩阵
    def get_result(self):
        for i in range(self.Mat_R):
            self.Mat_P_result[i, self.Mat_P[i]] = 1
            for j in range(self.Mat_C):
                if i > j:
                    self.Mat_L[i, j] = self.Mat[i, j]
                else:
                    self.Mat_U[i, j] = self.Mat[i, j]
        #print("P:", "\n", self.Mat_P_result)
        #print("L:", "\n", self.Mat_L)
        #print("U:", "\n", self.Mat_U)
        return self.Mat_P_result, self.Mat_L, self.Mat_U

    # 进行矩阵求逆，当detail=True时输出运行过程，以便进行bug定位
    # 求逆过程为逐行求逆 类比于Ax=b，x为A逆的每一列，b为单位阵I的每一列
    def get_inversion(self, detail=True):
        B = np.eye(self.Mat_R)
        for i in range(self.Mat_R):
            sub_B = np.dot(self.Mat_P_result, B[:, i])
            y = np.zeros([self.Mat_R, 1], dtype=np.float32)
            y[0] = sub_B[0]
            for m in range(self.Mat_R - 1):
                sum_y = 0
                for n in range(m + 1):
                    sum_y += self.Mat_L[m + 1, n] * y[n]
                y[m + 1] = sub_B[m + 1] - sum_y

            if detail:
                print("此时y为：", y, "\n")
                print("此时的乘积Ly为", "\n", np.dot(self.Mat_L, y))

            x = np.zeros([self.Mat_R, 1], dtype=np.float32)
            x[self.Mat_R - 1] = y[self.Mat_R - 1] / self.Mat_U[self.Mat_R - 1, self.Mat_R - 1]
            for p in range(self.Mat_R - 1):
                sum_x = 0
                for q in range(p + 1):
                    sum_x += self.Mat_U[-p - 2, -q - 1] * x[-q - 1]
                x[-p - 2] = (y[-p - 2] - sum_x) / self.Mat_U[-p - 2, -p - 2]

            if detail:
                print("此时x为：", x, "\n")
                print("此时的乘积Ux为", "\n", np.dot(self.Mat_U, x))
                print("此时的乘积Ax为", "\n", np.dot(self.Mat_stable, x))

            self.Mat_inversion[:, i] = x.reshape(self.Mat_R)
        print("逆矩阵为：", "\n", self.Mat_inversion)

        return self.Mat_inversion


def InvFunc(mat):
    # 创建对象
    test = PLU(mat)
    print("PLU")
    # 分解矩阵，当detail=True时输出分解过程
    test.Factorization(detail=False)
    print("Factorization")
    # 输出分解后的P,L,U矩阵
    P,L,U = test.get_result()
    print("test.get_result")
    # 进行矩阵求逆，当detail=True时输出运行过程，以便进行bug定位
    Inversion = test.get_inversion(detail=False)
    print("Inversion")
    return Inversion
