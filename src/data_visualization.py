import numpy as np
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names, n_attr
from scipy.stats import norm
import seaborn as sns

def print_histograms(data_matrix, class_labels):
    for i, attribute in enumerate(attributes_names):
        plt.figure()
        plt.xlabel(attribute)
        plt.title("Histogram distribution of attribute " + attribute)
        for current_class_label in range(len(class_names)):
            mask = (class_labels == current_class_label)
            plt.hist(data_matrix[i,mask], bins = 20, density = True, ec = 'black', alpha = 0.5)
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
                

def gaussianize_features(D):
    gauss_feat = np.zeros(D.shape)
    for feat in range(D.shape[0]):
        for sample in range(D.shape[1]):
            gauss_feat[feat, sample] = norm.ppf((sum(D[feat, sample] < D[feat, :]) + 1) / (D.shape[1] + 2))
    return gauss_feat


def feat_heatmap(D):
    sns.heatmap(np.corrcoef(D))
    plt.show()
    
def Z_score(D):
    return (D - D.mean(1).reshape((D.shape[0], 1))) / (np.var(D, axis = 1).reshape((D.shape[0], 1)) ** 0.5)


if __name__ == "__main__":


    # load the dataset
    data_matrix, class_labels = load("../Data/Train.txt")
    """
    print("*********** Some statistics about the dataset **************")
    print("Bad quality samples: %d, Good quality samples: %d" % (sum(class_labels == 0), sum(class_labels == 1)))

    gauss_feat = np.load("gaussianized_features.npy")
    feat_heatmap(gauss_feat)
    feat_heatmap(gauss_feat[:, class_labels == 0])
    feat_heatmap(gauss_feat[:, class_labels == 1])
    feat_heatmap(data_matrix)
    feat_heatmap(data_matrix[:, class_labels == 0])
    feat_heatmap(data_matrix[:, class_labels == 1])

    # gauss_feat = gaussianize_features(data_matrix)
    # np.save("gaussianized_features.npy", gauss_feat)
    """


    # print histograms 
    #print_histograms(data_matrix, class_labels)

    # print scatter plots
    print_scatterplots(data_matrix, class_labels)
    print(data_matrix.mean(1))
    print(np.var(data_matrix, axis = 1))
    # subtract mean
    #normalized_data_matrix = data_matrix - data_matrix.mean(1).reshape((data_matrix.shape[0], 1))
    #print_histograms(normalized_data_matrix, class_labels)
    #print_scatterplots(normalized_data_matrix, class_labels)
