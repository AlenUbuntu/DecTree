import sys
import os
from treeTraining import decisionTree
from RandomTreeTraining import randDecisionTree
from Validation import validation
from PruneTree import prune

def main():

    # check if the command form is correct
    if len(sys.argv)<6:
        print(" Correct Input Form:\n --python3--|--main.py--|--number of nodes to prune--|--training set path--|--validation set path --|--test set path--|--print or not--|")
        sys.exit("Error: Missing Arguments!\n exited").gettrace()
    # set home directory as the  PATH
    PATH = './Data/'
    # read arguments from the command
    pruneNum = int(sys.argv[1])
    trainingPath = PATH+sys.argv[2]
    validationPath = PATH+sys.argv[3]
    testPath = PATH+sys.argv[4]
    printTreeVal = int(sys.argv[5])
 
    # set Information entropy threshold to be 0, under which we regard it as a pure node
    threshold = 0

     
    # create a directory storing output data files
    os.makedirs('./Data/output',exist_ok=True)
    
    ########################     IG-based Tree          ################################
    ####################################################################################
    # build a decision tree and train it
    decisiontree = decisionTree(trainingPath,threshold)
    
    # file store the tree structure before pruning
    filehandle = open('./Data/output/original_tree.txt','w')
    
    # print tree
    decisiontree.printTree(filehandle) 
      
    # close file
    filehandle.close()
    
    #####################################################################################
    # file storing the accuracy information of the tree before pruning
    filehandle = open('./Data/output/accuracy.txt','a')
    filehandle.seek(0)
    filehandle.truncate()
    #####################################################################################
    
    # check accuracy with validation data set before pruning
    treeValid = validation(testPath,decisiontree.getTree()) 
    
    # print the accuracy on test data set
    treeValid.printAccuracy(filehandle)
    
    # print the average depth and total number of nodes
    decisiontree.printLevNod(filehandle)
    
    
    # prune the tree
    if pruneNum != 0:
        prunetree = prune(decisiontree,validationPath,pruneNum,0)
        prunetree.pruneTree()
        
        prunetree.printAccuracy(testPath,filehandle)
        
        # print the average depth and total number of nodes
        decisiontree.printLevNod(filehandle)
        filehandle.close()
        if printTreeVal == 1:
            ##############################################################################
            # create a file to store the IG-built tree after post-prune
            file = open('./Data/output/postprune_tree.txt','w')
            ##############################################################################
            prunetree.printTree(file) 
            ##############################################################################
            file.close()              
    
    ######################        Random-built Tree            #########################
    ####################################################################################
    # create a file storing the tree structure 
    filehandle = open('./Data/output/rand_original_tree.txt','w')
    
    # build a decision tree and train it by randomly selecting attributes
    decisiontree = randDecisionTree(trainingPath,threshold)
    
    
    # print tree
    decisiontree.printTree(filehandle) 
      
    # close file
    filehandle.close()
    
    #####################################################################################
    # file storing the accuracy information of the tree before pruning
    filehandle = open('./Data/output/rand_accuracy.txt','a')
    filehandle.seek(0)
    filehandle.truncate()
    #####################################################################################
    
    # check accuracy with validation data set before pruning
    treeValid = validation(testPath,decisiontree.getTree()) 
    
    # print the accuracy on test data set
    treeValid.printAccuracy(filehandle)
    
    # print the average depth and total number of nodes
    decisiontree.printLevNod(filehandle)
    
    # prune the tree
    if pruneNum != 0:
        prunetree = prune(decisiontree,validationPath,pruneNum,0)
        prunetree.pruneTree()
        
        prunetree.printAccuracy(testPath,filehandle)
        
        # print the average depth and total number of nodes
        decisiontree.printLevNod(filehandle)
        filehandle.close()
        
        if printTreeVal == 1:
            ##############################################################################
            # create a file to store the tree after post-prune
            file = open('./Data/output/rand_postprune_tree.txt','w')
            ##############################################################################
            prunetree.printTree(file)      
            ##############################################################################
            file.close()            
    
    
# Launch the program
if __name__ == '__main__':
    main()