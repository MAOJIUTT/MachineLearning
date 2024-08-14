import numpy as np
import pandas as pd
from cv2.ml import LogisticRegression
from sklearn import linear_model  # 调用sklearn的线性回归包



path = 'ex2data2.txt'
data = pd.read_csv(path, header=None, names=['test 1', 'test 2', 'accepted'])

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

model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y.ravel())
LogisticRegression()
print(model.score(X, y))