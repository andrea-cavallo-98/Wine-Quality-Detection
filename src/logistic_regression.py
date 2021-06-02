import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import min_DCF, compute_llr

def logreg_obj_wrap(DTR, LTR, l, pi_T):

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]

        Nt = sum(LTR == 1)
        Nf = sum(LTR == 0)
        J = l/2 * np.linalg.norm(w)**2 + pi_T / Nt * sum(np.log1p(np.exp( - (np.dot(w.T, DTR[:, LTR == 1]) + b )))) + \
            (1 - pi_T) / Nf * sum(np.log1p(np.exp((np.dot(w.T, DTR[:, LTR == 0]) + b ))))

        return J

    return logreg_obj


def binary_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp):
    
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, pi_T)

    optV, _, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = True)   

    w, b = optV[0:-1], optV[-1]

    # Compute scores
    s = np.dot(w.T, DTE) + b
    #PredictedLabels = np.zeros(DTE.shape[1])
    #PredictedLabels[s > 0] = 1
    #acc = sum(PredictedLabels == LTE) / LTE.shape[0]

    minDCF = min_DCF(s, pi, Cfn, Cfp, LTE)

    return s, minDCF


def k_fold_cross_validation(D, L, classifier, k, pi, Cfp, Cfn, l, pi_T, seed = 0):

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

        llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
        start_index += elements

    minDCF = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF



if __name__ == "__main__":

    # BINARY LOGISTIC REGRESSION

    # print("\n\n BINARY LOGISTIC REGRESSION\n\n")

    D, L = load("../Data/Train.txt")  
    (DTR, LTR), (DTE, LTE) = split_db_4to1(D, L)
    l_val = [0, 1e-6, 1e-3, 1]
    pi_T = 0.5
    pi = 0.5
    Cfn = 1
    Cfp = 1
    k = 5

    for l in l_val:
        lformat = "{:2e}".format(l)
        print("\n************** Lambda: %s ***************" % lformat)
        #minDCF = k_fold_cross_validation(D, L, binary_logistic_regression, k, pi, Cfp, Cfn, l, pi_T, seed = 0)
        _, minDCF = binary_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
        print("\nmin DCF: %.4f\n" % (minDCF))


