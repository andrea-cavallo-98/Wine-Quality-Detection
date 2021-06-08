import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import act_DCF, min_DCF, confusion_matrix, Bayes_risk
import support_vector_machines as SVM
from data_visualization import Z_score
import logistic_regression as LR
import gmm as GMM


def optimal_threshold(llr, L):

    pi = 0.5
    Cfn = 1
    Cfp = 1
    (DTR, LTR), (DTE, LTE) = split_db_4to1(llr.reshape([llr.shape[0], 1]), L)
    # Estimate optimal threshold on training split
    _, opt_t = min_DCF(DTR.reshape([DTR.shape[0],]), pi, Cfn, Cfp, LTR)

    # Estimate min DCF on test split
    minDCF, _ = min_DCF(DTE.reshape([DTE.shape[0],]), pi, Cfn, Cfp, LTE)

    # Estimate DCF using optimal threshold on test split
    PredictedLabels = np.zeros([DTE.shape[0]])
    PredictedLabels[DTE.reshape([DTE.shape[0],]) > opt_t] = 1
    M = confusion_matrix(LTE, PredictedLabels, 2)
    _, actDCF = Bayes_risk(M, pi, Cfn, Cfp)

    # Estimate DCF using optimal theoretical threshold on test split
    actDCFth = act_DCF(DTE.reshape([DTE.shape[0],]), pi, Cfn, Cfp, LTE)

    return opt_t, minDCF, actDCF, actDCFth


def combine_scores(llr1, llr2, L):

    pi = 0.5
    Cfn = 1
    Cfp = 1
    # Split llr
    (DTR1, LTR1), (DTE1, LTE1) = split_db_4to1(llr1.reshape([llr1.shape[0], 1]), L)
    (DTR2, LTR2), (DTE2, LTE2) = split_db_4to1(llr2.reshape([llr2.shape[0], 1]), L)

    # Define DTR and DTE as combinations of llr from different models
    DTR = np.zeros([2, DTR1.shape[0]])
    DTR[0, :] = DTR1.reshape([DTR1.shape[0],])
    DTR[1, :] = DTR2.reshape([DTR2.shape[0],])
    DTE = np.zeros([2, DTE1.shape[0]])
    DTE[0, :] = DTE1.reshape([DTE1.shape[0],])
    DTE[1, :] = DTE2.reshape([DTE2.shape[0],])

    # linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp):
    all_minDCF = []
    all_actDCF = []
    for l in [0, 1e-6, 1e-4, 1e-2, 1, 100]:
        s, minDCF = LR.linear_logistic_regression(DTR, LTR1, DTE, LTE1, l, 0.5, pi, Cfn, Cfp)
        actDCF = act_DCF(s, pi, Cfn, Cfp, LTE1)
        all_minDCF.append(minDCF)
        all_actDCF.append(actDCF)

    return all_minDCF, all_actDCF



if __name__ == "__main__":

    D, L = load("../Data/Train.txt")    
    DN = Z_score(D)
    k = 5
    pi = 0.5
    Cfp = 1
    Cfn = 1
    pi_T = 0.5

    # Compute actual DCF for most promising model assuming theoretical threshold
    # RBF kernel SVM with C = 10 and log(gamma) = -2, rebalacing and Z-normalized features
    _, llrSVM = SVM.k_fold_cross_validation(DN, L, SVM.kernel_SVM, k, pi, Cfp, Cfn, 10, pi_T, 1, rebalancing = True, type = "RBF", gamma = np.exp(-2), just_llr=True)
    """
    # Estimate optimal threshold on the llr
    opt_t, minDCF, actDCF, actDCFth = optimal_threshold(llrSVM, L)

    print(" \n*** RBF-SVM *** \n")
    print("min DCF on test set: " + str(minDCF))
    print("actual DCF on test set with optimal theoretical threshold: " + str(actDCFth))
    print("min DCF on test set with optimal estimated threshold: " + str(actDCF))
    print("optimal estimated threshold was: " + str(opt_t))
    """
    # Quadratic LR, lambda = 0, Z-normalized features, no PCA
    #_, llrLR = LR.k_fold_cross_validation(DN, L, LR.quadratic_logistic_regression, k, pi, Cfp, Cfn, 0, pi_T, seed = 0)
    #_, llrGMM = GMM.k_fold_cross_validation(D, L, k, pi, Cfp, Cfn, False, False, 8, seed = 0, just_llr = True)
    """
    # Estimate optimal threshold on the llr
    opt_t, minDCF, actDCF, actDCFth = optimal_threshold(llrLR, L)

    print(" \n*** Quad-LR *** \n")
    print("min DCF on test set: " + str(minDCF))
    print("actual DCF on test set with optimal theoretical threshold: " + str(actDCFth))
    print("min DCF on test set with optimal estimated threshold: " + str(actDCF))
    print("optimal estimated threshold was: " + str(opt_t))
    """

    # Train a linear logistic regression model to combine the scores of the two models 
    minDCF, actDCF = combine_scores(llrSVM, llrSVM, L)

    print("\n *** COMBINED SCORES *** \n")
    print("min DCF: " + str(minDCF))
    print("actual DCF: " + str(actDCF))
