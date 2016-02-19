from csvParser import csvParser
import copy
import math
import sys
import os

class treeNode:
    
    def __init__(self,label=-1,attr=-1,left=None,right=None,isLeaf=0):
       
        self.label = label                   # class label
        self.attr = attr                     # attribute index of this node
        self.isLeaf = isLeaf                 # isLeaf = 1 if this node is selected for prune or it is original leaf-> means it is leaf now
        self.left = left                     # left child
        self.right = right                   # right child

        
class decisionTree:
    
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
            
            self.tree = self.ID3(self.dataIndex,self.attrIndex,self.classLabelIndex,threshold)
            
            
        def ID3(self,dataIndex,attrIndex,classLabelIndex,threshold):
            
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
                
            # otherwise, choose the best attribute based on Information Gain
            else:
                
                # get best attribute
                bestAttribute = self.getBestAttribute(nodeEntropy,dataIndex,attrIndex,classLabelIndex)
                
                # set bestAttribute as this node's attribute
                root.attr = bestAttribute
                
                # copy the attribute list but remove the bestAttribute from the list
                attribute = copy.deepcopy(attrIndex)
                attribute.remove(bestAttribute)
                
                # For each training data, we determine which branch should it take
                # hence split the training data set based on the attribute we choose
                branch = self.split(dataIndex,bestAttribute)
                
                # construct child nodes of this tree
                
                root.left = self.ID3(branch[0][0],attribute,branch[0][1],threshold)
                root.right = self.ID3(branch[1][0],attribute,branch[1][1],threshold)
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
            
        def getBestAttribute(self,nodeEntropy,dataIndex,attrIndex,classLabelIndex):
            
            # define default maxGain 
            maxGain = -1
            # define attribute index that gives the maximum entropy
            maxIndex = -1
            
            # iterate over all attributes, calculate information gain and find the maximum gain
            for eachIndex in attrIndex:
                gain = self.getInformationGain(nodeEntropy,dataIndex,classLabelIndex,eachIndex)
                if gain > maxGain:
                    maxGain = gain
                    maxIndex = eachIndex
                
            return maxIndex
        
        def getInformationGain(self,nodeEntropy, dataIndex,classLabelIndex,attribute):
            
            # define four counters to count number of data in each case
            # attribute :0 and classLabel :0  - Negneg
            # attribute :0 and classLabel :1  - Negpos
            # attribute :1 and classLabel :0  - Posneg
            # attribute :1 and classLabel :1  - Pospos
            Negneg = 0
            Negpos = 0
            Posneg = 0
            Pospos = 0
            
            for i in dataIndex:
                if self.dataset[i][attribute]==0:
                    if self.classLabel[i] == 0:
                        Negneg += 1
                    else:
                        Negpos += 1
                else:
                    if self.classLabel[i] == 0:
                        Posneg += 1
                    else:
                        Pospos += 1
            
            # calculate p_mj ^i 
            if Negneg !=0:
                NegnegFraction = Negneg / (Negneg + Negpos)
            else:
                NegnegFraction = 0
                
            NegposFraction = 1 - NegnegFraction
            
            if Posneg != 0:
                PosnegFraction = Posneg / (Posneg + Pospos)
            else:
                PosnegFraction = 0
                
            PosposFraction = 1 - PosnegFraction
            
            # calculate p_j
            NegFraction = (Negneg + Negpos)/len(dataIndex)
            PosFraction = (Posneg + Pospos)/len(dataIndex)
            
            entropy = 0
            # calculate entropy
            if NegnegFraction == 0:
                t1 = 0
            else:
                t1 = NegnegFraction * math.log2(NegnegFraction)
                
            if NegposFraction == 0:
                t2 = 0
            else:
                t2 = NegposFraction * math.log2(NegposFraction)
                
            entropy -= NegFraction * (t1+ t2)
            
            if PosnegFraction == 0:
                t1 = 0
            else:
                t1 = PosnegFraction * math.log2(PosnegFraction)
                
            if PosposFraction == 0:
                t2 = 0
            else:
                t2 = PosposFraction * math.log2(PosposFraction)
            entropy -= PosFraction * (t1+ t2)
            
            return nodeEntropy - entropy
        
        def split(self,dataIndex,bestAttribute):
            
            # create four empty list to hold data and class labels
            data_attr0 = []
            data_attr1 = []
            class_attr0 = []
            class_attr1 = []
            
            for i in dataIndex:
                if self.dataset[i][bestAttribute] == 0:
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
            
                    
            
                    
                
            
                    
            
            
                    
            
            
    
            
            
                
            
            
            
            