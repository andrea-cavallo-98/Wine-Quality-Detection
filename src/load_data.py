"""

*** LOAD AND SPLIT DATA ***

Functions to load the dataset and to split it when needed (two splits, one containing 80% of the data, 
the other 20%), plus some information on the data (such as attribute names, number of classes...)

"""

import numpy as np

# Information about the dataset
attributes_names = ["Fixed acidity", "Volatile acidity", "Citric acidity", "Residual sugar", "Chlorides",
                     "Free sulfur dioxide", "Total sulfur dioxide", "Density", "pH", "Sulphates", "Alcohol"]
n_attr = len(attributes_names)
class_names = ["Low quality", "High quality"]
n_class = len(class_names)

# Load dataset from textual file in a specified format (one sample per line, 
# features are comma-separated)
def load(fileName):
    
    class_labels_list = []
    list_of_vectors = []
    
    with open(fileName) as f:
        for line in f:
            try:
                current_vector = np.array(line.split(",")[0:n_attr], dtype = np.float).reshape((n_attr,1))
                list_of_vectors.append(current_vector)
                class_labels_list.append(int (line.split(",")[n_attr] ))
            except:
                pass

    data_matrix = np.array(list_of_vectors).reshape((len(list_of_vectors),n_attr)).T
    class_labels = np.array(class_labels_list)

    return data_matrix, class_labels

# Split the data in two parts (80%-20%)
def split_db_4to1(D, L, seed=0):
    np.random.seed(seed)

    if(D.shape[1] == 1):
        nTrain = int(D.shape[0]*4.0/5.0)
        idx = np.random.permutation(D.shape[0])
    else:    
        nTrain = int(D.shape[1]*4.0/5.0)
        idx = np.random.permutation(D.shape[1])
    
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    if(D.shape[1] == 1):
        DTR = D[idxTrain]
        DTE = D[idxTest]
    else:
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)

