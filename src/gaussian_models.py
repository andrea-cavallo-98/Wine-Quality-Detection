import numpy as np
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
import numpy.linalg
from scipy.special import logsumexp
from prediction_measurement import min_DCF, compute_llr
from pca import compute_pca
from data_visualization import Z_score

def mean_and_covariance_matrix(D):

    # find dataset mean
    mu = D.mean(1).reshape((D.shape[0],1))
    # center the data
    DC = D - mu
    # compute covariance matrix
    C = np.dot(DC, DC.T) / D.shape[1]
    return mu, C


def GAU_logpdf_ND(x, mu, C):
    M = x.shape[0]
    _, logSigma = numpy.linalg.slogdet(C)
    
    if x.shape[1] == 1:
        logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.dot(np.dot((x - mu).T, numpy.linalg.inv(C)), (x-mu))
    else:
        logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.diagonal(np.dot(np.dot((x - mu).T, numpy.linalg.inv(C)), (x-mu)))

    return logN


def MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs = True):

    mu = []
    sigma = []
    for c in range(n_class):
        m, s = mean_and_covariance_matrix(DTR[:, LTR == c])
        mu.append(m)
        sigma.append(s)
    
    S = np.zeros([n_class, DTE.shape[1]])
    
    if use_logs:

        for c in range(n_class):
            # log domain
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], sigma[c])
        
        # Compute joint probabilities
        SJoint = S + np.log(1/n_class)
        # Compute marginal log-densities
        SMarg = logsumexp(SJoint, axis = 0)
        # Compute class log-posterior probabilities
        SPost = SJoint - SMarg

    else:

        for c in range(n_class):
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], sigma[c])

        # Compute joint probabilities
        SJoint = S / n_class
        # Compute class posterior probabilities
        SPost = SJoint / SJoint.sum(axis = 0)

    llr = compute_llr(SPost)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF


def naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs = True):

    mu = []
    sigma = []
    for c in range(n_class):
        m, s = mean_and_covariance_matrix(DTR[:, LTR == c])
        mu.append(m)
        sigma.append(s * np.eye(s.shape[0]))

    S = np.zeros([n_class, DTE.shape[1]])
    
    if use_logs:

        for c in range(n_class):
            # log domain
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], sigma[c])

        # Compute joint probabilities
        SJoint = S + np.log(1/n_class)
        # Compute marginal log-densities
        SMarg = logsumexp(SJoint, axis = 0)
        # Compute class log-posterior probabilities
        SPost = SJoint - SMarg

    else:

        for c in range(n_class):
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], sigma[c])

        # Compute joint probabilities
        SJoint = S / n_class
        # Compute class posterior probabilities
        SPost = SJoint / SJoint.sum(axis = 0)

    llr = compute_llr(SPost)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF



def tied_MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs = True):

    mu = []
    tied_sigma = np.zeros([DTR.shape[0], DTR.shape[0]])

    for c in range(n_class):
        m, s = mean_and_covariance_matrix(DTR[:, LTR == c])
        mu.append(m)
        tied_sigma += sum(LTR == c) / DTR.shape[1] * s
        
    S = np.zeros([n_class, DTE.shape[1]])
    
    if use_logs:
            
        for c in range(n_class):
            # log domain
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], tied_sigma)

        # Compute joint probabilities
        SJoint = S + np.log(1/n_class)
        # Compute marginal log-densities
        SMarg = logsumexp(SJoint, axis = 0)
        # Compute class log-posterior probabilities
        SPost = SJoint - SMarg

    else:

        for c in range(n_class):
            # log domain
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], tied_sigma)

        # Compute joint probabilities
        SJoint = S / n_class
        # Compute class posterior probabilities
        SPost = SJoint / SJoint.sum(axis = 0)

    llr = compute_llr(SPost)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF


def tied_naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs = True):

    mu = []
    tied_sigma = np.zeros([DTR.shape[0], DTR.shape[0]])

    for c in range(n_class):
        m, s = mean_and_covariance_matrix(DTR[:, LTR == c])
        mu.append(m)
        tied_sigma += sum(LTR == c) / DTR.shape[1] * ( s * np.eye(s.shape[0])) 
        
    S = np.zeros([n_class, DTE.shape[1]])
    
    if use_logs:
            
        for c in range(n_class):
            # log domain
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], tied_sigma)

        # Compute joint probabilities
        SJoint = S + np.log(1/n_class)
        # Compute marginal log-densities
        SMarg = logsumexp(SJoint, axis = 0)
        # Compute class log-posterior probabilities
        SPost = SJoint - SMarg

    else:

        for c in range(n_class):
            # log domain
            S[c,:] = GAU_logpdf_ND(DTE, mu[c], tied_sigma)

        # Compute joint probabilities
        SJoint = S / n_class
        # Compute class posterior probabilities
        SPost = SJoint / SJoint.sum(axis = 0)

    llr = compute_llr(SPost)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF


def k_fold_cross_validation(D, L, classifier, k, pi, Cfp, Cfn, seed = 0):

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

    llr = np.zeros([D.shape[1], ])

    for count in range(k):

        if start_index + elements > D.shape[1]:
            end_index = D.shape[1]
        else:
            end_index = start_index + elements 

        idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
    
        DTE = D[:, idxTest]
        LTE = L[idxTest]

        llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, pi, Cfn, Cfp)
        start_index += elements

    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF


if __name__ == "__main__":
    D, L = load("../Data/Train.txt")  
    D = Z_score(D)
    (DTR, LTR), (DTE, LTE) = split_db_4to1(D, L)
    DG = np.load("gaussianized_features.npy")
    (DGTR, LGTR), (DGTE, LGTE) = split_db_4to1(DG, L)
    DG10 = compute_pca(10, DG)
    (DGTR10, LGTR10), (DGTE10, LGTE10) = split_db_4to1(DG10, L)
    DG9 = compute_pca(9, DG)
    (DGTR9, LGTR9), (DGTE9, LGTE9) = split_db_4to1(DG9, L)
    k = 5
    use_logs = True
    pi = 0.5
    Cfn = 1
    Cfp = 1
    fileName = "../Results/gaussian_resultsZ.txt"

    with open(fileName, "w") as f:
    
        f.write("*** min DCF for different gaussian models ***\n\n")
        f.write("Gaussianized features - no PCA\n")
        f.write("5-fold cross validation\n")
        # K FOLD CROSS VALIDATION
        DCF_MVG = k_fold_cross_validation(DG, L, MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_naive_Bayes = k_fold_cross_validation(DG, L, naive_Bayes, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_MVG = k_fold_cross_validation(DG, L, tied_MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_naive_Bayes = k_fold_cross_validation(DG, L, tied_naive_Bayes, k, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
                " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        f.write("\nsingle fold\n")
        # K FOLD CROSS VALIDATION
        _, DCF_MVG = MVG(DGTR, LGTR, DGTE, LGTE, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = naive_Bayes(DGTR, LGTR, DGTE, LGTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = tied_MVG(DGTR, LGTR, DGTE, LGTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = tied_naive_Bayes(DGTR, LGTR, DGTE, LGTE, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
            " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))

        
        f.write("\n\nGaussianized features - PCA = 10\n")
        f.write("5-fold cross validation\n")
        # K FOLD CROSS VALIDATION
        DCF_MVG = k_fold_cross_validation(DG10, L, MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_naive_Bayes = k_fold_cross_validation(DG10, L, naive_Bayes, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_MVG = k_fold_cross_validation(DG10, L, tied_MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_naive_Bayes = k_fold_cross_validation(DG10, L, tied_naive_Bayes, k, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
            " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        f.write("\nsingle fold\n")
        # K FOLD CROSS VALIDATION
        _, DCF_MVG = MVG(DGTR10, LGTR10, DGTE10, LGTE10, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = naive_Bayes(DGTR10, LGTR10, DGTE10, LGTE10, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = tied_MVG(DGTR10, LGTR10, DGTE10, LGTE10, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = tied_naive_Bayes(DGTR10, LGTR10, DGTE10, LGTE10, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) 
            + " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        
        f.write("\n\nGaussianized features - PCA = 9\n")
        f.write("5-fold cross validation\n")
        # K FOLD CROSS VALIDATION
        DCF_MVG = k_fold_cross_validation(DG9, L, MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_naive_Bayes = k_fold_cross_validation(DG9, L, naive_Bayes, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_MVG = k_fold_cross_validation(DG9, L, tied_MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_naive_Bayes = k_fold_cross_validation(DG9, L, tied_naive_Bayes, k, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
            " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        f.write("\nsingle fold\n")
        # K FOLD CROSS VALIDATION
        _, DCF_MVG = MVG(DGTR9, LGTR9, DGTE9, LGTE9, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = naive_Bayes(DGTR9, LGTR9, DGTE9, LGTE9, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = tied_MVG(DGTR9, LGTR9, DGTE9, LGTE9, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = tied_naive_Bayes(DGTR9, LGTR9, DGTE9, LGTE9, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) 
            + " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))

        f.write("\n\nRaw features - no PCA\n")
        f.write("5-fold cross validation\n")
        # K FOLD CROSS VALIDATION
        DCF_MVG = k_fold_cross_validation(D, L, MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_naive_Bayes = k_fold_cross_validation(D, L, naive_Bayes, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_MVG = k_fold_cross_validation(D, L, tied_MVG, k, pi, Cfp, Cfn, use_logs)
        DCF_tied_naive_Bayes = k_fold_cross_validation(D, L, tied_naive_Bayes, k, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) 
            + " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        f.write("\nsingle fold\n")
        # K FOLD CROSS VALIDATION
        _, DCF_MVG = MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = tied_MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = tied_naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
            " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        

