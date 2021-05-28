import numpy as np
import matplotlib.pyplot as plt

attributes_names = ["Fixed acidity", "Volatile acidity", "Citric acidity", "Residual sugar", "Chlorides",
                     "Free sulfur dioxide", "Total sulfur dioxide", "Density", "pH", "Sulphates", "Alcohol"]
n_attr = len(attributes_names)
class_names = ["Low quality", "High quality"]


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


def print_histograms(data_matrix, class_labels):
    for i, attribute in enumerate(attributes_names):
        plt.figure()
        plt.xlabel(attribute)
        plt.title("Histogram distribution of attribute " + attribute)
        for current_class_label in range(len(class_names)):
            mask = (class_labels == current_class_label)
            plt.hist(data_matrix[i,mask], bins = 10, density = True, ec = 'black', alpha = 0.5)
        plt.legend(class_names)
        plt.show()    

def print_scatterplots(data_matrix, class_labels):
    for i, attribute1 in enumerate(attributes_names):
        for j, attribute2 in enumerate(attributes_names):
            if attribute1 != attribute2:
                plt.figure()
                plt.xlabel(attribute1)
                plt.ylabel(attribute2)
                plt.title("Scatterplot distribution of attributes " + attribute1 + " and " + attribute2)
                for current_class_label in range(len(class_names)):
                    mask = (class_labels == current_class_label)
                    plt.scatter(data_matrix[i,mask], data_matrix[j,mask])
                plt.legend(class_names)
                plt.show()
                


if __name__ == "__main__":


    # load the dataset
    data_matrix, class_labels = load("../Data/Train.txt")

    # print histograms 
    print_histograms(data_matrix, class_labels)

    # print scatter plots
    print_scatterplots(data_matrix, class_labels)
    
    # subtract mean
    normalized_data_matrix = data_matrix - data_matrix.mean(1).reshape((data_matrix.shape[0], 1))
    print_histograms(normalized_data_matrix, class_labels)
    print_scatterplots(normalized_data_matrix, class_labels)
