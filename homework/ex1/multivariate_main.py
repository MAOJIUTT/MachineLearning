import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def computeCost(X, y, theta):
    J = np.power(((X * theta.T) - y), 2)
    return np.sum(J) / (2 * len(X))


def batchGradientDescent(X, y, theta, alpha, iterations):
    # 创建一个与 theta 形状相同的零矩阵
    temp = np.matrix(np.zeros(theta.shape))
    # 获取 theta 的参数数量
    parameters = int(theta.ravel().shape[1])
    # 初始化一个用于存储迭代次数的成本数组
    cost = np.zeros(iterations)


    for i in range(iterations):
        loss = (X * theta.T) - y

        # 遍历所有参数，并更新
        for j in range(parameters):
            # 偏导数的一部分
            term = np.multiply(loss, X[:,j])
            # 更新theta
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# 归一化
data2 = (data2 - data2.mean()) / data2.std()


data2.insert(0, 'Ones', 1)
alpha = 0.01
iterations = 1000

cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]        # 前三列作为x，因为房子大小和卧室数量是影响因素
y2 = data2.iloc[:, cols-1:cols]     # 最后一列作为y，因为我们最终是要预测房价

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.zeros((1, 3)))

result2_theta, cost2 = batchGradientDescent(X2, y2, theta2, alpha, iterations)

computeCost(X2, y2, result2_theta)

# 绘制拟合函数的3D图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

x_1 = np.linspace(data2.Size.min(), data2.Size.max(), 100)
x_2 = np.linspace(data2.Bedrooms.min(), data2.Bedrooms.max(), 100)

# 将 x_1 和 x_2 转换为网格矩阵 X 和 Y, X 和 Y 都是 100x100 的二维数组。
X, Y = np.meshgrid(x_1, x_2)

# 二维的结果
fitting_function2_2d = result2_theta[0, 0] + (result2_theta[0, 1] * X) + (result2_theta[0, 2] * Y)

# 绘制三维曲面图，绘制三维图像需要二维数据
surface = ax.plot_surface(X, Y, fitting_function2_2d, cmap='viridis', alpha=0.7)

# 一维的结果
fitting_function2_1d = result2_theta[0, 0] + (result2_theta[0, 1] * x_1) + (result2_theta[0, 2] * x_2)

# 绘制三维散点图，绘制三维散点需要一维数据
scatter = ax.scatter(x_1, x_2, fitting_function2_1d, c='r', marker='o')

# 设置标签
ax.set_title('predicted profit')
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

plt.show()

# 绘制loss函数图像
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iterations), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()