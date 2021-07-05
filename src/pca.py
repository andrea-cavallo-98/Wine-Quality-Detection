"""

*** PCA ***

Functions to compute PCA over a dataset, both for training and for test data 
(which require to project the test data on the dimensions evaluated from the training data).

"""

import numpy as np

def covariance_matrix(D):
    # find dataset mean
    mu = D.mean(1).reshape((D.shape[0],1))
    # center the data
    DC = D - mu
    # compute covariance matrix
    C = np.dot(DC, DC.T) / D.shape[1]
    return C

# Compute PCA on the given data 
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

# Compute PCA on training data and project test data on the
# calculated directions
def compute_pca_eval(m, DTR, DTE, use_svd = False):
    C = covariance_matrix(DTR)
    # compute eigenvalues and eigenvectors
    if use_svd:
        U, _, _ = np.linalg.svd(C)
        P = U[:, 0:m]
    else:
        _, U = np.linalg.eigh(C)
        # extract leading eigenvectors
        P = U[:, ::-1][:, 0:m]
    # apply projection to initial matrix
    DP = np.dot(P.T, DTE)
    return DP

