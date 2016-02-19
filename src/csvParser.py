import csv

class csvParser:
    def __init__(self, path):
      
        # create an object list to store the data read from a csv file
        self.filepath = path
        self.dataset = []
        self.attrset = []
        self.classLabel = []
        self.attrIndex = []
        self.dataIndex = []
        self.classLabelIndex = []
        self.read()


    def read(self):   
        
        count = 0
        # retrieve attributes name and attribute list for each instance
        
        try:
            csvfile = open(self.filepath,'r')
            try:
                print('\n',self.filepath,'is successfully opened!\n\n Start parsing ...')
                csvreader = csv.reader(csvfile, delimiter = ',')
                for row in csvreader:
                    if count == 0:
                        # retrieve the attribute names from the header of the file
                        self.attrset = row[:len(row)-1]
                    else:
                        # retrieve the attribute list for each training data
                        self.dataset.append([int(i) for i in row[:len(row)-1]])
                        # retrieve the class label for each training data
                        self.classLabel.append(int(row[-1]))
                    count += 1
                
                # create corresponding index array for attributes and training data
                self.attrIndex = list(range(len(self.attrset)))
                self.dataIndex = list(range(len(self.dataset)))
                self.classLabelIndex = list(range(len(self.classLabel)))
            except EOFError:
                print(' Critical Error in reading the file')
            finally:
                csvfile.close()
                print(' Done\n opened file is successfully closed!')
        except OSError:
            print(' Error: file specified is not found')
            
               
                    
                    
                