# gradients 
- 目录内为梯度下降算法相关代码和结果
### 1. 文件组成
- forwarding.py 为各前向传播函数
- BP.py 为各反向传播函数
- 运行 main.py 进行梯度下降
### 2. 调试注意事项
- 更改三个文件的 sampler 可以更改降采样频率
  - 必须同时更改
  - sampler 较小时会有内存超限报错且耗时较长
- 打开 main.py 中相关注释可以实现
  - 可视化三维点云
  - 单独储存各个方向的法向量图
  - 输出各个点 E12 E345 E6 的和，反映各项E的变化
- 可以通过更改 step 数组改变每次迭代步长
### 3. results 目录
- I.jpg 为原始灰度图，mask.jpg 为排除边缘点演示图
- B 目录存储各次迭代重构灰度图
- normal 目录存储各次迭代重构法向量图
- res 目录存储各次迭代 I 和 B 之间的残差图
### 4. 存在问题
- 耗时过长，具体体现在梯度矩阵相乘环节
  - 可以考虑稀疏性
- albedo 统一当成 1, 可以考虑加入albedo
- 梯度下降应当用减号却用加号才有下降效果
  - 需要对梯度进行检验 
### 5.修改
- forward：
  - line42:去掉/fx,/fy
  - line122：p3[2][n]=D[n]
  - line180:int强制转换报错？
- BP：
  - line123，line124感觉应该缩进
- main：
  - line135：改成`dloss_dD=np.dot(np.dot(np.dot(np.dot(dloss_dE,dE_dB),dB_dH),dH_dn),dn_dD)+np.dot(np.dot(dloss_dE,dE_dp),dp_dD)+np.dot(dloss_dE,dE_dD)`，也许会算得更快
### 6. 12.28调整
- 做出了第五项的修改
  - 耗时大大减小
  - step 较小时可以实现梯度下降
  - 调整超参数`[w1, w2, w3] = [1, 400, 1000]`保证三项约束量级相同
- 运行试验
  - 保证不发生内存超限，只能最低选取`sampler = 5`
  - 步长 0.0000001，循环 250 次，运行时间约 7 小时
  - 第 199 次产生最小 loss 为 2920605.54291
- 新增文件
  - `./results/res/3d.png` 三维点云图
  - `loss.png`三种约束loss的下降曲线
  - `loss_show.png`loss变化曲线
  - `results.txt`存储运行结果：循环次数，运行时间，以及每次的各项 loss
- 仍存在问题
  - 数值微分的检验
  - 得到的B直观上并没有产生太大平滑