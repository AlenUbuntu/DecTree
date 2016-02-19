from csvParser import csvParser
import copy
import math
import random
import sys
import os

class treeNode:
    
    def __init__(self,label=-1,attr=-1,left=None,right=None,isLeaf=0):
       
        self.label = label                   # class label
        self.attr = attr                     # attribute index of this node
        self.isLeaf = isLeaf                 # isLeaf = 1 if this node is selected for prune or it is original leaf-> means it is leaf now
        self.left = left                     # left child
        self.right = right                   # right child

        
class randDecisionTree:
    
        def __init__(self,filepath,threshold):
            
            print("start training the tree...")
            
            # create a csvParser object which parses the csv file specified by the 
            # filepath, return data collected from the file.          
            
            csvhandle = csvParser(filepath)
            
            # get associated data
            self.dataset = csvhandle.dataset
            self.attrset = csvhandle.attrset
            self.classLabel = csvhandle.classLabel
            self.attrIndex = csvhandle.attrIndex
            self.dataIndex = csvhandle.dataIndex
            self.classLabelIndex = csvhandle.classLabelIndex
            self.counter = 0
            self.dcounter = 0
            self.lcounter = 0
            
            self.tree = self.RandomSelect(self.dataIndex,self.attrIndex,self.classLabelIndex,threshold)
            
            
        def RandomSelect(self,dataIndex,attrIndex,classLabelIndex,threshold):
            
            # if training data set is empty, return none
           
            if len(dataIndex)==0:
                return None
            
            # create a root node
            root = treeNode()
         
            # get the entropy of this particular node
            nodeEntropy = self.getNodeEntropy(classLabelIndex)
         
            # get the majority class label
            root.label = self.getMajorityClass(classLabelIndex)
            
            # if the dataset is already pure (determined by threshold) or there is no attribute available, 
            # return the node
            if nodeEntropy <= threshold or len(attrIndex)==0:
                root.isLeaf = 1
                return root
                
            # otherwise, randomly select a attribute based on which we split the datasets
            else:
                
                # get random attribute
                randomAttr = attrIndex[random.randint(0,len(attrIndex)-1)]
                
                # set random Attribute as this node's attribute
                root.attr = randomAttr
                
                # copy the attribute list but remove the random Attribute from the list
                attribute = copy.deepcopy(attrIndex)
                attribute.remove(randomAttr)
                
                # For each training data, we determine which branch should it take
                # hence split the training data set based on the attribute we choose
                branch = self.split(dataIndex,randomAttr)
                
                # construct child nodes of this tree
                
                root.left = self.RandomSelect(branch[0][0],attribute,branch[0][1],threshold)
                root.right = self.RandomSelect(branch[1][0],attribute,branch[1][1],threshold)
                return root
            
        def getNodeEntropy(self,classLabelIndex):
            
            # set negative example counter
            negativeCounter = 0
            
            for each in classLabelIndex:
                if self.classLabel[each] == 0:
                    negativeCounter += 1
                 
            # calculate fraction
            # positiveFraction and negativeFraction 
            negativeFraction = negativeCounter / len(classLabelIndex)
            positiveFraction = 1 - negativeFraction
            
            # calculate entropy
            if positiveFraction == 0 or negativeFraction == 0:
                return 0
            else:
                return - positiveFraction * math.log2(positiveFraction) - negativeFraction * math.log2(negativeFraction)
                
        def getMajorityClass(self,classLabelIndex):
            
            count = 0
            # count positive samples
            for each in classLabelIndex:
                if self.classLabel[each] == 1:
                    count += 1
            
            if count >= (len(classLabelIndex)-count):
                return 1
            else:
                return 0
            
        
        def split(self,dataIndex,randomAttribute):
            
            # create four empty list to hold data and class labels
            data_attr0 = []
            data_attr1 = []
            class_attr0 = []
            class_attr1 = []
            
            for i in dataIndex:
                if self.dataset[i][randomAttribute] == 0:
                    data_attr0.append(i)
                    class_attr0.append(i)
                else:
                    data_attr1.append(i)
                    class_attr1.append(i)
            
            return [[data_attr0,class_attr0],[data_attr1,class_attr1]]
        
        def getTree(self):
            return self.tree
        
        # public printTree method, can be called outside the class
        def printTree(self,filehandle): 
            sys.stdout = filehandle        
            self.__printTree__(self.tree,-1,-1,self.tree.attr)
            sys.stdout = sys.__stdout__
        
        # private printTree method, should not be invoked outside the class
        def __printTree__(self,root,level,val,attr): 
            
            # set up the string needed to print in this line
            stri = ""
            # concatenate leading indent in each line
            stri = stri + "| " * level
            
            # concatenate the following content in the line
            
            # if root is NUll, simply return ,do nothing
            if root == None:
                return 
            
            # if it is root, do nothing
            if val == -1:
                pass
            else:
                if root.isLeaf == 1:
                    stri = stri + str(self.attrset[attr]) +' = ' + str(val) +' : '+ str(root.label)
                else:
                    stri = stri + str(self.attrset[attr]) +' = ' + str(val) +' : '
                    
            print (stri)
            self.__printTree__(root.left,level+1,0,root.attr)
            self.__printTree__(root.right,level+1,1,root.attr)
            
            
        def printLevNod(self,filehandle):
            sys.stdout = filehandle  
            self.counter = 0
            self.dcounter = 0
            self.lcounter = 0
            self.__countTree__(self.tree,-1,-1)
            print("number of nodes is: ", self.counter)
            print("average depth is: ", self.dcounter/self.lcounter)
            sys.stdout = sys.__stdout__
            
        # private countTree method, should not be invoked outside the class
        def __countTree__(self,root,level,val): 
            
            # if root is NUll, simply return ,do nothing
            if root == None:
                return 
            
            # if it is root, do nothing
            if val == -1:
                pass
            else:
                if root.isLeaf == 1:
                    self.dcounter += level
                    self.lcounter += 1
                    
            self.counter += 1
            self.__countTree__(root.left,level+1,0)
            self.__countTree__(root.right,level+1,1)
                    
            
                    
                
            