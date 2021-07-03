import numpy as np
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names
from scipy.stats import norm
import seaborn as sns

# Save histograms with distribution of the features of the dataset
def print_histograms(data_matrix, class_labels, figName):
    for i, attribute in enumerate(attributes_names):
        plt.figure()
        plt.xlabel(attribute)
        plt.title("Histogram distribution of attribute " + attribute)
        for current_class_label in range(len(class_names)):
            mask = (class_labels == current_class_label)
            plt.hist(data_matrix[i,mask], bins = 20, density = True, ec = 'black', alpha = 0.5)
        plt.legend(class_names)
        plt.savefig("../Images/" + figName + str(i)+".png")   
        plt.close() 
                
# Compute feature gaussianization
def gaussianize_features(D):
    gauss_feat = np.zeros(D.shape)
    for feat in range(D.shape[0]):
        for sample in range(D.shape[1]):
            gauss_feat[feat, sample] = norm.ppf((sum(D[feat, sample] < D[feat, :]) + 1) / (D.shape[1] + 2))
    return gauss_feat

# Apply Gaussianization to test data using training data as a reference
def gaussianize_features_eval(DTR, DTE):
    gauss_feat = np.zeros(DTE.shape)
    for feat in range(DTE.shape[0]):
        for sample in range(DTE.shape[1]):
            gauss_feat[feat, sample] = norm.ppf((sum(DTE[feat, sample] < DTR[feat, :]) + 1) / (DTR.shape[1] + 2))
    return gauss_feat

# Save the heatmap of correlation among features
def feat_heatmap(D, figName):
    plt.figure()
    sns.heatmap(np.corrcoef(D))
    plt.savefig("../Images/" + figName + ".png")
    plt.close()
    
# Compute Z-score normalization (center data and normalize variance)
def Z_score(D):
    return (D - D.mean(1).reshape((D.shape[0], 1))) / (np.var(D, axis = 1).reshape((D.shape[0], 1)) ** 0.5)

# Apply Z-score normalization to test data using mean and variance of training dataset 
def Z_score_eval(DTR, DTE):
    return (DTE - DTR.mean(1).reshape((DTR.shape[0], 1))) / (np.var(DTR, axis = 1).reshape((DTR.shape[0], 1)) ** 0.5)


if __name__ == "__main__":

    ### Load the dataset
    data_matrix, class_labels = load("../Data/Train.txt")
    
    print("*********** Some statistics about the dataset **************")
    print("Bad quality samples: %d, Good quality samples: %d" % (sum(class_labels == 0), sum(class_labels == 1)))

    ### Distributions of features 
    print_histograms(data_matrix, class_labels, "RawFeatHist")
    
    ### Gaussianize features and save them
    # gauss_feat = gaussianize_features(data_matrix)
    # np.save("gaussianized_features.npy", gauss_feat)
    gauss_feat = np.load("../Data/gaussianized_features.npy")
    print_histograms(gauss_feat, class_labels, "GaussFeatHist")

    ### Analyze correlations among features
    feat_heatmap(gauss_feat, "GaussFeatHeat")
    feat_heatmap(gauss_feat[:, class_labels == 0], "GaussFeatHeat0")
    feat_heatmap(gauss_feat[:, class_labels == 1], "GaussFeatHeat1")
    feat_heatmap(data_matrix, "RawFeatHeat")
    feat_heatmap(data_matrix[:, class_labels == 0], "RawFeatHeat0")
    feat_heatmap(data_matrix[:, class_labels == 1], "RawFeatHeat1")

    ### Gaussianize test data using training data as reference
    DTR, LTR = load("../Data/Train.txt")
    DTE, LTE = load("../Data/Test.txt")
    # DTE_gauss = gaussianize_features_eval(DTR, DTE)
    # np.save("gaus_test.npy", DTE_gauss)
    DTE_gauss = np.load("../Data/gaus_test.npy")

    ### Distributions of gaussianized features
    print_histograms(DTE_gauss, LTE, "GaussFeatHist")






