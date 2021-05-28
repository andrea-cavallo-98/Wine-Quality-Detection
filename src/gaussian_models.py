import numpy as np
import matplotlib.pyplot as plt
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_2to1
import numpy.linalg
from scipy.special import logsumexp


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


def MVG(DTR, LTR, DTE, LTE, use_logs = True):

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

    PredictedLabels = SPost.argmax(axis = 0)
    acc = sum(PredictedLabels == LTE) / DTE.shape[1]
    n_err = sum(PredictedLabels != LTE)
    return PredictedLabels, acc, n_err, SPost


def naive_Bayes(DTR, LTR, DTE, LTE, use_logs = True):

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

    PredictedLabels = SPost.argmax(axis = 0)
    acc = sum(PredictedLabels == LTE) / DTE.shape[1]
    n_err = sum(PredictedLabels != LTE)
    return PredictedLabels, acc, n_err, SPost



def tied_covariance(DTR, LTR, DTE, LTE, use_logs = True):

    mu = []
    sigma = []
    for c in range(n_class):
        m, s = mean_and_covariance_matrix(DTR[:, LTR == c])
        mu.append(m)
        sigma.append(s)

    tied_sigma = np.zeros(sigma[0].shape)
    for c in range(n_class):
        tied_sigma += sum(LTR == c) * sigma[c]
    tied_sigma /= DTR.shape[1]
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

    PredictedLabels = SPost.argmax(axis = 0)
    acc = sum(PredictedLabels == LTE) / DTE.shape[1]
    n_err = sum(PredictedLabels != LTE)
    return PredictedLabels, acc, n_err, SPost


def k_fold_cross_validation(D, L, classifier, k, seed = 0):

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)
    tot_err = 0

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

        _, _, err, _ = classifier(DTR, LTR, DTE, LTE)
        tot_err += err
        start_index += elements

    return (1 - tot_err / D.shape[1])


if __name__ == "__main__":
    D, L = load("../Data/Train.txt")  
    k = 10
    use_logs = True

    # K FOLD CROSS VALIDATION
    acc_MVG = k_fold_cross_validation(D, L, MVG, k, use_logs)
    acc_naive_Bayes = k_fold_cross_validation(D, L, naive_Bayes, k, use_logs)
    acc_tied_covariance = k_fold_cross_validation(D, L, tied_covariance, k, use_logs)
    
    print("MVG: Accuracy = %.4f, Error rate = %.4f " % (acc_MVG, 1-acc_MVG))
    print("Naive Bayes: Accuracy = %.4f, Error rate = %.4f " % (acc_naive_Bayes, 1-acc_naive_Bayes))
    print("Tied covariance: Accuracy = %.4f, Error rate = %.4f " % (acc_tied_covariance, 1-acc_tied_covariance))





