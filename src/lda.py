
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names, n_attr, n_class


def vrow(vet):
    return vet.reshape((1,vet.shape[0]))

def covariance_matrix(D):
    # find dataset mean
    mu = D.mean(1).reshape((D.shape[0],1))
    # center the data
    DC = D - mu
    # compute covariance matrix
    C = np.dot(DC, DC.T) / D.shape[1]
    return C

def within_between_covariances(D, class_labels):
    # dataset mean
    mu = D.mean(1).reshape((D.shape[0],1))
    # compute between class covariance and within class covariance 
    
    SB = np.zeros((D.shape[0],D.shape[0]))
    SW = np.zeros((D.shape[0],D.shape[0]))
    
    for current_class in range(n_class):
        current_class_dataset = D[:, class_labels == current_class]
        n_c = current_class_dataset.shape[0]
        mu_c = current_class_dataset.mean(1).reshape((D.shape[0],1))
        SB += np.dot(mu_c - mu, (mu_c - mu).T)

        C = covariance_matrix(current_class_dataset)
        SW += C

    SB = SB / D.shape[1]
    SW = SW / D.shape[1]

    return SB, SW


def compute_lda(m, D, class_labels, generalized_eig = True):
    # compute between and within covariances
    SB, SW = within_between_covariances(D, class_labels)
    # find LDA directions

    if generalized_eig:
        s, U = scipy.linalg.eigh(SB, SW)
        W = U[:, ::-1][:, 0:m]
        
    else:
        U,s,_ = np.linalg.svd(SW)
        P1 = np.dot(U * vrow(1.0/(s**0.5)), U.T)
        #P1 = np.dot(np.dot(U, np.diag(1.0/(s**0.5))), U.T)
        SBT = np.dot(np.dot(P1, SB), P1.T)
        s, U = np.linalg.eigh(SBT)
        P2 = U[:, ::-1][:, 0:m]
        W = np.dot(P1.T, P2)

    # find a basis of W
    UW, _, _ = np.linalg.svd(W)
    W_basis = UW[:, 0:m]

    return np.dot(W.T, D)



if __name__ == "__main__":
    data_matrix, class_labels = load("../Data/Train.txt")
    projected_matrix = compute_lda(2, data_matrix, class_labels, True)
    for current_class_label in range(n_class):
        mask = (class_labels == current_class_label)
        plt.scatter(projected_matrix[0,mask], projected_matrix[1,mask])
    plt.legend(class_names)
    plt.show()


