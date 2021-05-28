import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_2to1


def logreg_obj_wrap(DTR, LTR, l):

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]

        J = l/2 * np.linalg.norm(w)**2 + 1/DTR.shape[1] * sum(np.log1p(np.exp( - (2*LTR - 1) * (np.dot(w.T, DTR) + b ))))

        return J

    return logreg_obj


def binary_logistic_regression(DTR, LTR, DTE, LTE, l):
    
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)

    optV, optJ, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = True)   

    w, b = optV[0:-1], optV[-1]

    # Compute scores
    s = np.dot(w.T, DTE) + b
    PredictedLabels = np.zeros(DTE.shape[1])
    PredictedLabels[s > 0] = 1
    acc = sum(PredictedLabels == LTE) / LTE.shape[0]

    return optJ, 1-acc


if __name__ == "__main__":

    # BINARY LOGISTIC REGRESSION

    # print("\n\n BINARY LOGISTIC REGRESSION\n\n")

    D, L = load("../Data/Train.txt")  
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    l_val = [0, 1e-6, 1e-3, 1]

    for l in l_val:
        lformat = "{:2e}".format(l)
        print("\n************** Lambda: %s ***************" % lformat)
        optJ, err = binary_logistic_regression(DTR, LTR, DTE, LTE, l)
        optJformat = "{:2e}".format(optJ)
        print("\n Objective function: %s Error rate: %.4f\n" % (optJformat, err))


