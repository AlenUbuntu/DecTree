from Validation import validation 
from treeTraining import decisionTree
import random
import copy
import sys

class prune:
    def __init__(self,tree,filepath,nodeNum,levelcontrol):        
        
        self.tree = tree                   # training tree passed 
        self.treeroot = self.tree.getTree()# root of this tree
        self.filepath = filepath
        self.nodeNum = nodeNum             # number of nodes needed to prune 
        self.treeList = []
        self.levelcontrol = levelcontrol   # prune start from this node
        
    
    def treeToList(self,root): 
        self.treeList = []
        self.__treeToList__(root)
        
    
    # Internal method, should not be called outside class   
    def __treeToList__(self,root):
        if self.tree == None:
            raise ReferenceError("empty tree!")
        
        if root == None:
            return
        
        self.treeList.append(root)
        self.__treeToList__(root.left)
        self.__treeToList__(root.right)
    
    def randomNumberGenerator(self):
            
        while True:
            # generate random numbers starting from the given node number
            tmp = random.randint(self.levelcontrol, len(self.treeList)-1)
            
            # check if this number is root or if its corresponding node is a leaf node. If so, ignore this number
            if self.treeList[tmp].isLeaf == 1 or tmp == 0:
                pass
            else:
                return tmp
        
   
                
    def pruneTree(self):
        
        # iterator_stop:
        stop = 100
        
        # record attributes about max accuracy
        self.maxAttr = []
         
        # set bestTree to original tree
        bestTree = self.treeroot   
        
        # create validation handler
        bestvalidat = validation(self.filepath,bestTree)  
        validat = validation(self.filepath,None)
        
        # calculate default max accuracy
        maxaccu = bestvalidat.calculateAccuracy()
        
        for i in range(stop):
            
            # copy current best tree to currentTree for further operation
            currentTree = copy.deepcopy(bestTree)           
            validat.setTreeRoot(currentTree)
            
            # set up a temporary list to record deleted nodes
            tmp = []
            
            for n in range(self.nodeNum):
                
                # convert currentTree to a list of tree nodes
                self.treeToList(currentTree)
                     
                if len(self.treeList) <= 3:
                    break
                # generate random Numbers
                j = self.randomNumberGenerator() 
                self.treeList[j].isLeaf = 1
                self.treeList[j].left = None 
                self.treeList[j].right = None
                tmp.append(self.treeList[j])
           
            newaccu = validat.calculateAccuracy()
            if newaccu > maxaccu:
                self.maxAccuracy = newaccu
                maxaccu = newaccu
                bestTree = currentTree

        self.maxAttr = [self.tree.attrset[i.attr] for i in tmp]
        
        # reset the original training tree to best tree
        self.tree.tree = bestTree
        self.treeroot = bestTree
        
        
    def printTree(self,filehandle):
        # print tree
        self.tree.printTree(filehandle)
    
    def printAccuracy(self,testPath,filehandle):  
        tmp = validation(testPath,self.treeroot)
        sys.stdout = filehandle
        print('\n|-----------------------------------------------------------------------------------------------|')
        print('\n |--number of nodes tried to prune: ',self.nodeNum,'--|--starting node number: ',self.levelcontrol,' --|')
        print('\n Node selected to prune: ',self.maxAttr)
        print(' The prediction accuracy on given data set after pruning the tree is ',format(tmp.calculateAccuracy(),'5.2%'))
        print('|-----------------------------------------------------------------------------------------------|\n')    
        sys.stdout = sys.__stdout__
            
        
    