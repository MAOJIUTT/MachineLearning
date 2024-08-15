from numpy import *
import operator

def classify0(inX, dataSet, labels, k):
    # inX代表一个（1，n）的向量
    dataSetSize = dataSet.shape[0]
    # 生成和dataset一样的形状的矩阵，进行向量运算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 按列相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 从小到大排序，但是返回值为索引值
    sortedDistIndicies = distances.argsort()    # 索引值列表
    classCount = {}
    # 计算标签的数量
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # classCount.items()是将其转化为dict_items([(key1, value1), (key2, value2), ...])的(key, value)形式
    # operator.itemgetter(1)是一个函数，它从每个元组中提取第一个索引（1，即值）作为排序的关键字。这意味着排序是基于字典的值进行的，而不是键。
    # reverse = True参数表示按降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回最大数量的标签
    return sortedClassCount[0][0]


def creatDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

if __name__ == '__main__':
    group, labels = creatDataSet()
    inX = array([1.0, 2.0])
    print(classify0(inX, group, labels, 3))