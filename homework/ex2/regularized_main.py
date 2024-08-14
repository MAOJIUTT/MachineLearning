import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as opt

def logistic(z):
    return 1 / (1 + np.exp(-z))

# *args 的用法在后面会说明
def cost(theta, X, y, lr, *args):
    theta = np.array(theta)
    X = np.array(X)
    y = np.array(y)

    h = logistic(X @ theta)
    return (-np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) + (np.mean(np.power(theta, 2)) * lr / 2))

def gradient(theta, X, y, lr, alpha):
    theta = np.array(theta).reshape(-1, 1)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    m = len(X)

    error = logistic(X @ theta) - y

    grad = alpha * (X.T @ error) / m

    reg_term = (lr / m) * theta
    reg_term[0] = 0

    grad += reg_term

    return grad.flatten()

def predict(theta, X):
    probability = logistic(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]



path = 'ex2data2.txt'
data = pd.read_csv(path, header=None, names=['test 1', 'test 2','accepted'])


# positive = data[data['accepted'].isin([1])]
# negative = data[data['accepted'].isin([0])]
#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['test 1'], positive['test 2'], c='b', marker='o', label='Accepted')
# ax.scatter(negative['test 1'], negative['test 2'], c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()


degree = 5
x1 = data['test 1']
x2 = data['test 2']

data.insert(3, 'ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data.drop('test 1', axis=1, inplace=True)
data.drop('test 2', axis=1, inplace=True)


cols = data.shape[1]
X = data.iloc[:,1:cols]
y = data.iloc[:,0:1]

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(11)

lr = 1
print(cost(theta, X, y, lr))

alpha = 1
print(gradient(theta, X, y, lr, alpha))

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y, lr, alpha))
print(result)

theta_min = np.array(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) / len(correct))

print(accuracy)