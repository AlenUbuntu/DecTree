# DecTree
Decision Tree Implementation Based on Information Gain Metric (Machine Learning)


DecTree
      DecTree is a machine learning program that implements decision tree algorithm in order to learn boolean data sets. 
      It supports csv file parsing and tree post-prune.

Table of contents:
      Installation and Program running
      File List

(I) Installation and Program running:

To run this python program, you need python 3.4.X installed on your PC. It
should be emphasized here that python 2.x is incompatible with python 3.4.X,
and hence may cause some problem.

Open a terminal, go to the directory where you store main.y, (Linux) Type:

   python3 main.py N training_set_path validation_set_path test_set_path C

   1. training_set_path, validation_set_path, test_set_path are paths relevant
      to 'Data/' directory. All the training/validation/test data sets should
      be put in the Data directory. For example:

      if you put these tree data sets in
      'Data/data_sets1/training_set.csv'

      the path entered in command should be
      '/data_sets1/training_set.csv'

    2. N - number of tree nodes you want to prune
    3. C - boolean control signal (0/1) that notifies if the tree after pruning
           should be printed out

 The program will print the prediction accuracy before and after post pruning

Note:
    To modify number of iteration cycles for tree pruning, change line 52 in prune.py(default value = 100)
    To print the decision tree before pruning, uncomment line 38 in main.py

(II) File List

    main.py               ----  launcher program that implements output directory creation, class creation, function call, etc

    csvParser.py          ----  a python class which parses the csv file provided by the given file path, throw any exception or error met

    decisionTree.py       ----  a python class that implements decision tree learning algorithm, choosing attributes based maximum information gain

    PruneTree.py          ----  a python class that implements post-prune algorithm by randomly selecting a node and prune the subtrees below this node (simply record the accuracy without really pruning tree). Trying this multiple times, select the one gives maximum accuracy and prune tree accordingly as the final tree

    Validation.py         ----  a python class that calculate the accuracy of a DT for the given data set

    /data_sets1
          training_set.csv
          validation_set.csv
          test_set.csv

    /data_sets2
          training_set.csv
          validation_set.csv
          test_set.csv

    /output                         ---- generated by the program
          accuracy.txt              ---- accuracy report before pruning and after pruning tree
          original_tree.txt         ---- tree structure for training tree before pruning
          postprune_tree.txt        ---- tree structure after pruning
          rand_accuracy.txt         ---- accuracy report before pruning and after pruning tree (constructed by randomly selecting an attribute)
          rand_original_tree.txt    ---- tree structure for training tree before pruning (constructed by randomly selecting an attribute)
          rand_postprune_tree.txt   ---- tree structure after pruning (constructed by randomly selecting an attribute)

    /analysis
        /result1
            summary.txt             ---- summary report of accuracy of decision tree on test data set in data_sets1/, 5 different N values are used
            tree_before_prune.txt   ---- original decision tree before pruning
            tree_after_prune.txt    ---- best decision tree after pruning in these five trials
            trees/                  ---- folder containing the tree structures before and after pruning for these five trials

        /result2
            summary.txt             ---- summary report of accuracy of decision tree on test data set in data_sets2/, 5 different N values are used
            tree_before_prune.txt   ---- original decision tree before pruning
            tree_after_prune.txt    ---- best decision tree after pruning in these five trials
            trees/                  ---- folder containing the tree structures before and after pruning for these five trials
