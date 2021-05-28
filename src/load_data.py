
import numpy as np

attributes_names = ["Fixed acidity", "Volatile acidity", "Citric acidity", "Residual sugar", "Chlorides",
                     "Free sulfur dioxide", "Total sulfur dioxide", "Density", "pH", "Sulphates", "Alcohol"]
n_attr = len(attributes_names)
class_names = ["Low quality", "High quality"]
n_class = len(class_names)

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


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

