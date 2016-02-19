from treeTraining import decisionTree
from csvParser import csvParser
import sys


class validation:
    def __init__(self,filepath,root):
        
        # parse the validation csv file
        csvhandle = csvParser(filepath)
        
        # get data set and corresponding class label for each data
        self.dataset = csvhandle.dataset
        self.classLabel = csvhandle.classLabel
        
        # set the training tree
        self.treeroot = root
        
    def getPredictedValue(self,root,data):
        
        # if tree does not exist, raise an Error
        if self.treeroot == None:
            raise ReferenceError("training tree does not exist or is not successfully passed to validation set")
        
        # if root is none, return -1
        if root == None:
            return -1
        
        # if root is a leaf node, return its class label
        if root.isLeaf == 1:
            return root.label
        
        # if attribute value of this data is 0, go to left child
        if data[root.attr] == 0: 
            return self.getPredictedValue(root.left, data)
        
        # if attribute value of this data is 1, go to right child
        else:
            return self.getPredictedValue(root.right, data)
    
    def calculateAccuracy(self):
        
        # if validation data set is not successfully passed or the validation data set is empty, return 0 
        if self.dataset == None or len(self.dataset) == 0:
            raise IOError('validation data set is empty!')
        
        # correct predict counter
        correctCounter  = 0
        
        for i in range(len(self.classLabel)):
            predictVal = self.getPredictedValue(self.treeroot, self.dataset[i])
            if predictVal == self.classLabel[i]:
                correctCounter += 1
        
        self.accuracy = correctCounter / len(self.classLabel)
        return self.accuracy
    
    def printAccuracy(self,filehandle):
            sys.stdout = filehandle
            accu = self.calculateAccuracy()
            print('\n|-----------------------------------------------------------------------------------------------|')
            print(' The prediction accuracy on given data set without pruning the tree is ',format(accu,'5.2%'))
            print('|-----------------------------------------------------------------------------------------------|\n')
            sys.stdout=sys.__stdout__
        
    def setTreeRoot(self,treeroot):
        self.treeroot = treeroot

            