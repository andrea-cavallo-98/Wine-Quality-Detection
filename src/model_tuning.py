import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import act_DCF, min_DCF, confusion_matrix, Bayes_risk
import support_vector_machines as SVM
from data_visualization import Z_score
import logistic_regression as LR
import gmm as GMM


S = 6

def optimal_threshold(llr, L):

    pi = 0.5
    Cfn = 1
    Cfp = 1
    (DTR, LTR), (DTE, LTE) = split_db_4to1(llr.reshape([llr.shape[0], 1]), L, seed = S)
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



def score_calibration(llr, L):
    
    pi = 0.5
    Cfn = 1
    Cfp = 1
    # Split llr
    (DTR, LTR), (DTE, LTE) = split_db_4to1(llr.reshape([llr.shape[0], 1]), L, seed = S)

    all_minDCF = []
    all_actDCF = []
    for l in [0, 1e-5, 1e-3, 0.1, 1, 10, 100]:
        s, minDCF = LR.linear_logistic_regression(DTR.reshape([1, DTR.shape[0]]), LTR, DTE.reshape([1, DTE.shape[0]]), LTE, l, 0.5, pi, Cfn, Cfp, calibration=True)
        actDCF = act_DCF(s, pi, Cfn, Cfp, LTE)
        all_minDCF.append(minDCF)
        all_actDCF.append(actDCF)
    

    return all_minDCF, all_actDCF



def combine_scores3(llr1, llr2, llr3, L):

    pi = 0.5
    Cfn = 1
    Cfp = 1
    # Split llr
    (DTR1, LTR1), (DTE1, LTE1) = split_db_4to1(llr1.reshape([llr1.shape[0], 1]), L, seed = S)
    (DTR2, LTR2), (DTE2, LTE2) = split_db_4to1(llr2.reshape([llr2.shape[0], 1]), L, seed = S)
    (DTR3, LTR3), (DTE3, LTE3) = split_db_4to1(llr3.reshape([llr3.shape[0], 1]), L, seed = S)

    # Define DTR and DTE as combinations of llr from different models
    DTR = np.zeros([3, DTR1.shape[0]])
    DTR[0, :] = DTR1.reshape([DTR1.shape[0],])
    DTR[1, :] = DTR2.reshape([DTR2.shape[0],])
    DTR[2, :] = DTR3.reshape([DTR3.shape[0],])

    DTE = np.zeros([3, DTE1.shape[0]])
    DTE[0, :] = DTE1.reshape([DTE1.shape[0],])
    DTE[1, :] = DTE2.reshape([DTE2.shape[0],])
    DTE[2, :] = DTE3.reshape([DTE3.shape[0],])

    
    # linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp):
    all_minDCF = []
    all_actDCF = []
    for l in [0, 1e-6, 1e-4, 1e-2, 1, 100]:
        s, minDCF = LR.linear_logistic_regression(DTR, LTR1, DTE, LTE1, l, 0.5, pi, Cfn, Cfp, calibration=False)
        actDCF = act_DCF(s, pi, Cfn, Cfp, LTE1)
        all_minDCF.append(minDCF)
        all_actDCF.append(actDCF)

    return all_minDCF, all_actDCF



def combine_scores(llr1, llr2, L):

    pi = 0.5
    Cfn = 1
    Cfp = 1
    # Split llr
    (DTR1, LTR1), (DTE1, LTE1) = split_db_4to1(llr1.reshape([llr1.shape[0], 1]), L, seed = S)
    (DTR2, LTR2), (DTE2, LTE2) = split_db_4to1(llr2.reshape([llr2.shape[0], 1]), L, seed = S)

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
        s, minDCF = LR.linear_logistic_regression(DTR, LTR1, DTE, LTE1, l, 0.5, pi, Cfn, Cfp, calibration=False)
        actDCF = act_DCF(s, pi, Cfn, Cfp, LTE1)
        all_minDCF.append(minDCF)
        all_actDCF.append(actDCF)

    return all_minDCF, all_actDCF


def analyze_scores(llr, L, name):
    # Estimate optimal threshold on the llr
    opt_t, minDCF, actDCF, actDCFth = optimal_threshold(llr, L)
    # Perform score calibration using linear log reg
    minDCFcal, actDCFcal = score_calibration(llr, L)

    print(" \n*** " + name +" *** \n")
    print("min DCF on test set: " + str(minDCF))
    print("actual DCF on test set with optimal theoretical threshold: " + str(actDCFth))
    print("actual DCF on test set with optimal estimated threshold: " + str(actDCF))
    print("min DCF on test set with score calibration: " + str(minDCFcal))
    print("actual DCF on test set with score calibration: " + str(actDCFcal))
    print("optimal estimated threshold was: " + str(opt_t))


if __name__ == "__main__":

    D, L = load("../Data/Train.txt")   
    DN = Z_score(D)
    k = 5
    pi = 0.5
    Cfp = 1
    Cfn = 1
    pi_T = 0.5
    (DNTR, LNTR), (DNTE, LNTE) = split_db_4to1(DN, L, seed = S)


    ##################### SELECTED MODELS #######################
    # RBF-SVM, Z-normalized features, C=10, loggamma=-2, rebalancing
    # Quadratic logistic regression, Z-normalized features, lambda = 0
    # GMM, Z-normalized features, 8 components


    ###### RBF kernel SVM with C = 10 and log(gamma) = -2, rebalacing and Z-normalized features ######
    
    #_, llrSVM = SVM.k_fold_cross_validation(DN, L, SVM.kernel_SVM, k, pi, Cfp, Cfn, 10, pi_T, 1, rebalancing = True, type = "RBF", gamma = np.exp(-2), just_llr=True)
    #np.save("llrSVM.npy", llrSVM)
    llrSVM = np.load("llrSVM.npy")
    #analyze_scores(llrSVM, L, "RBF-SVM")

    ###### Quadratic LR, lambda = 0, Z-normalized features, no PCA ######

    #_, llrLR = LR.k_fold_cross_validation(DN, L, LR.quadratic_logistic_regression, k, pi, Cfp, Cfn, 0, pi_T)
    #np.save("llrLR.npy", llrLR)
    llrLR = np.load("llrLR.npy")
    #analyze_scores(llrLR, L, "Log Reg")
   
    ###### GMM, Z-normalized features, 8 components ######
    #_, llrGMM = GMM.k_fold_cross_validation(DN, L, k, pi, Cfp, Cfn, False, False, 8, seed = 0, just_llr = True)
    #np.save("llrGMM.npy", llrGMM)
    llrGMM = np.load("llrGMM.npy")
    #analyze_scores(llrGMM, L, "GMM")



    ###### Combined models ######
    
    minDCF, actDCF = combine_scores(llrSVM, llrGMM, L)

    print("\n *** COMBINED SCORES SVM + GMM *** \n")
    print("min DCF: " + str(minDCF))
    print("actual DCF: " + str(actDCF))
    
        
    minDCF, actDCF = combine_scores(llrSVM, llrLR, L)

    print("\n *** COMBINED SCORES SVM + LR *** \n")
    print("min DCF: " + str(minDCF))
    print("actual DCF: " + str(actDCF))
    
    minDCF, actDCF = combine_scores3(llrSVM, llrLR, llrGMM, L)

    print("\n *** COMBINED SCORES SVM + LR + GMM *** \n")
    print("min DCF: " + str(minDCF))
    print("actual DCF: " + str(actDCF))