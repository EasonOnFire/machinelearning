# c4.5 core algorithm
# -*- coding:utf-8 -*-
from math import log

# calc shannon entropy of label or feature
# rows in dataSet like: [feat0, feat1, ..., label]
def calcShannonEntropyOfFeature(dataSet, feat):
	numEntries = len(dataSet)
	labelCounts = {}
	for feaVec in dataSet:
		currentLabel = feaVec[feat]
		if currentLabel not in labelCounts:
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannon
	
	
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatureVec = featVec[:axis]
			reducedFeatureVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatureVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1 # last col is label
	baseEntropy = calcShannonEntropyOfFeature(dataSet, -1)
	bestInfoGainRate = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [raw[i] for raw in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEntropyOfFeature(subDataSet, -1)
		infoGain = baseEntropy - newEntropy
		iv = calcShannonEntropyOfFeature(dataSet, i)
		if (iv == 0):
			continue
		infoGainRate = infoGain / iv
		if infoGainRate > bestInfoGainRate:
			bestInfoGainRate = infoGainRate
			bestFeature = id
	return bestFeature
	
def majorityCnt(classList):
	classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)    
	
def createTree(dataSet, labels):
	classList = [raw[-1] for raw in dataSet]
	if classList.count(classList[0]) == len(classList):	# all data belong to same label
		return classList[0]
	if len(dataSet[0]) == 1:	# all feature exhausted
		return majorityCnt(classList)
		
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	if(bestFeat == -1):	# if class isnt relative of features, randomly choose one result
		return classList[0]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [raw[bestFeat] for raw in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree
	