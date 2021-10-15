"""
实现决策树算法对于是否买电脑问题的学习
"""
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
# from sklearn.externals.six import StringIo
import sklearn

# 导入数据集
# allElectronicsData = open(r'D:\0_Program\pythonDemo\ML_AND_DL\DTree\AllElectronics.csv', 'rb')
# reader = csv.reader(allElectronicsData)

dataList = []
featureList = []
labelList = []

# 读取数据
with open(r'D:\0_Program\pythonDemo\ML_AND_DL\DTree\AllElectronics.csv', 'r') as allElectronicsData:
    reader = csv.reader(allElectronicsData)

    for key, row in enumerate(reader):
        # 获取表头信息
        if key == 0:
            headers = row

        # 存储标签信息
        if key != 0:
            labelList.append(row[len(row) - 1])

        # 存储特征值
        if key != 0:
            rowDict = {}
            for i in range(1, len(row) - 1):
                rowDict[headers[i]] = row[i]
            featureList.append(rowDict)

# 数据预处理
# 二值化
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

# print(vec.get_feature_names_out())
# print("dummyX" + str(dummyX))
# print("dummyY" + str(dummyY))

# 生成决策树算法生成分类器
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

# 可视化
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# oneRowX = dummyX[0, :]
# print("oneRowX" + str(oneRowX))
#
# newRowX = oneRowX
#
# newRowX[0] = 1
# newRowX[2] = 0
# print("newRowX: " + str(newRowX))
#
# predictedY = clf.predict(newRowX)
# print("predictedY" + str(predictedY))


