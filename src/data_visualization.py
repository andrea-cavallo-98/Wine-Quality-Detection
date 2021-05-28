import numpy as np
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names, n_attr

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
            if (attribute1 != attribute2):
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

    print("*********** Some statistics about the dataset **************")
    print("Bad quality samples: %d, Good quality samples: %d" % (sum(class_labels == 0), sum(class_labels == 1)))

    # print histograms 
    #print_histograms(data_matrix, class_labels)

    # print scatter plots
    print_scatterplots(data_matrix, class_labels)
    
    # subtract mean
    normalized_data_matrix = data_matrix - data_matrix.mean(1).reshape((data_matrix.shape[0], 1))
    print_histograms(normalized_data_matrix, class_labels)
    print_scatterplots(normalized_data_matrix, class_labels)
