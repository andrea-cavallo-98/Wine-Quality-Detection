
import numpy as np
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names, n_attr, n_class

def covariance_matrix(D):
    # find dataset mean
    mu = D.mean(1).reshape((D.shape[0],1))
    # center the data
    DC = D - mu
    # compute covariance matrix
    C = np.dot(DC, DC.T) / D.shape[1]
    return C


def compute_pca(m, D, use_svd = False):
    C = covariance_matrix(D)
    # compute eigenvalues and eigenvectors
    if use_svd:
        U, _, _ = np.linalg.svd(C)
        P = U[:, 0:m]
    else:
        _, U = np.linalg.eigh(C)
        # extract leading eigenvectors
        P = U[:, ::-1][:, 0:m]
    # apply projection to initial matrix
    DP = np.dot(P.T, D)
    return DP



if __name__ == "__main__":
    data_matrix, class_labels = load("../Data/Train.txt")
    projected_matrix = compute_pca(2, data_matrix)
    for current_class_label in range(n_class):
        mask = (class_labels == current_class_label)
        plt.scatter(projected_matrix[0,mask], projected_matrix[1,mask])
    plt.legend(class_names)
    plt.show()



