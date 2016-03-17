#本脚本利用kNN算法对iris数据集进行分类，以前10%的数据作为测试数据，后90%的数据作为样本数据集。
#由于iris数据集原本按照三种花的顺序排列，在测试时前10%全为第一种花，不能完整测试kNN的分类功能，
#故将后两种花随机挑选数据各5项放入前10%。后面的散点图将iris数据集的前三维数据画出。
from numpy import *
import operator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

#载入数据
def file2matrix():
    file = open('irisdata.txt')
    allLines = file.readlines()
    row = len(allLines)
    dataSet = zeros((row, 4))
    labels = []
    index = 0
    for line in allLines:
        line = line.strip()
        listFromLine = line.split(',')
        dataSet[index, :] = listFromLine[0:4]
        labels.append(listFromLine[-1])
        index += 1        
    return dataSet, labels
    
def kNN(x, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    distance1 = tile(x, (dataSetSize,1)) - dataSet
    distance2 = distance1 ** 2
    distance3 = distance2.sum(axis=1)
    distance4 = distance3 ** 0.5
    sortedIndex = distance4.argsort()
    classCount = {}
    for i in range (k):
        label = labels[sortedIndex[i]]
        classCount[label] = classCount.get(label,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def kNN_test():
    testRatio = 0.1
    dataSet, labels = file2matrix()
    row = dataSet.shape[0]
    testNum = int(row * testRatio)
    error = 0.0
    for i in range (testNum):
        result = kNN(dataSet[i, :], dataSet[testNum:row, :], labels[testNum:row], 3)
        print 'the result came back with: %s, the real answer is: %s' % (result, labels[i])
        if (result != labels[i]):
            error += 1.0
    print 'error rate is: %f' % (error/float(testNum))
    

kNN_test()
'''
#画出散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
type1 = ax.scatter(dataSet[0:50, 0], dataSet[0:50, 1], dataSet[0:50, 2], c='r', marker='^')
type2 = ax.scatter(dataSet[50:100, 0], dataSet[50:100, 1], dataSet[50:100, 2], c='b', marker='o')
type3 = ax.scatter(dataSet[100:150, 0], dataSet[100:150, 1], dataSet[100:150, 2], c='y', marker='x')
ax.legend((type1, type2, type3), ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
'''

























