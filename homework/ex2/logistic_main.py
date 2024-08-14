import numpy as np
import pandas as pd
import scipy.optimize as opt
from matplotlib import pyplot as plt

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


def logistic(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    theta = np.array(theta)
    X = np.array(X)
    y = np.array(y)

    h = logistic(X @ theta)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))


def gradient(theta, X, y):
    theta = np.array(theta)
    X = np.array(X)
    y = np.array(y)

    error = logistic(X @ theta) - y
    return (X.T @ error) / len(X)


# 方便计算h_theta(x)
data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# 转换为 NumPy 数组
X = np.array(X.values)
y = np.array(y.values).flatten()  # Ensure y is a 1D array
theta = np.zeros(3)  # Ensure theta is a 1D array

a = gradient(theta, X, y)
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)
print(a)