from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics


#初始化SparkContext
sc = SparkContext()



#從hdfs拿資料
rdd  = sc.textFile('hdfs://master:9000/user/hduser/test/homework/final.csv')

#拿掉header
header = rdd.first()
rwd = rdd.filter(lambda x:x !=header)

#分割資料
raa = rwd.map(lambda x:x.replace("\"",""))
lines = raa.map(lambda x:x.split(","))


#將資料轉成float
def convert_float(x):
	return (float(x))


#分出x
def extract_features(field):
	numerfeatures = [convert_float(field) for field in field[2:30]]
	return numerfeatures


#分出y
def extract_label(field):
	label = field[1]
	return float(label)

#將資料轉成lablepoint
labelpointRDD = lines.map(lambda r:LabeledPoint(extract_label(r),extract_features(r)))


#拆訓練集跟測試集
(trainingData, testData) = labelpointRDD.randomSplit([0.75, 0.25])

#訓練模型
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={})

#預測test data
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)


#計算accuracy,precision,recall
accuracy = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1]).count() / float(testData.count())
deathprecision = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1] and lp[0] == 1).count() / float(labelsAndPredictions.filter(lambda lp: lp[1] == 1).count())
deathrecall = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1] and lp[0] == 1).count() / float(labelsAndPredictions.filter(lambda lp: lp[0] == 1).count())	
aliveprecision = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1] and lp[0] == 0).count() / float(labelsAndPredictions.filter(lambda lp: lp[1] == 0).count())
aliverecall = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1] and lp[0] == 0).count() / float(labelsAndPredictions.filter(lambda lp: lp[0] == 0).count())





print('accuracy = ' + str(accuracy))
print('deathprecision = ' + str(deathprecision))
print('deathrecall = ' + str(deathrecall))
print('aliveprecision = ' + str(aliveprecision))
print('aliverecall = ' + str(aliverecall))
print(model.toDebugString())
