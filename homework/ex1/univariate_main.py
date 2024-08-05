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

path = 'ex1data1.txt'

# header代表标题行，为None时代表无标题行，为0则代表第一行为标题行
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 插入1列全为1的数据
data.insert(0,'ones',1)

# 变量初始化
cols = data.shape[1]            # cols=3
X = data.iloc[:, 0:cols-1]      # X代表选取所有行，并且0~2列（2不取）
y = data.iloc[:, cols-1:cols]   # y代表选取所有行，并且2~3列（3不取）

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.zeros((1, 2)))

alpha = 0.01
iterations = 1000
result_theta, cost = batchGradientDescent(X, y, theta, alpha, iterations)

# 由于已经知道了拟合函数的参数result_theta，所以只需要生成x的值然后带入得到函数值，最后将其传入生成图像

# 在一个范围内生成均匀分布的数组
x = np.linspace(data.Population.min(), data.Population.max(), 100)
fitting_function = result_theta[0, 0] + (result_theta[0, 1] * x)

# plt.subplots创建子图，子图允许在同一图表窗口中显示多个图形
# fig代表整个图表，ax代表子图，一般只对ax进行操作，因为fig可以看作全局设置
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, fitting_function, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)

# 在多图设置时，也就是存在子图时，最好使用set_xlabel，set_title等
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit and Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()