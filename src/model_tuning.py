import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import act_DCF, min_DCF, confusion_matrix, Bayes_risk
import support_vector_machines as SVM
from data_visualization import Z_score
import logistic_regression as LR
import gmm as GMM

"""
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

    # Train a linear logistic regression model using different values of lambda
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
    (DTR2, _), (DTE2, _) = split_db_4to1(llr2.reshape([llr2.shape[0], 1]), L, seed = S)
    (DTR3, _), (DTE3, _) = split_db_4to1(llr3.reshape([llr3.shape[0], 1]), L, seed = S)

    # Define DTR and DTE as combinations of llr from different models
    DTR = np.zeros([3, DTR1.shape[0]])
    DTR[0, :] = DTR1.reshape([DTR1.shape[0],])
    DTR[1, :] = DTR2.reshape([DTR2.shape[0],])
    DTR[2, :] = DTR3.reshape([DTR3.shape[0],])

    DTE = np.zeros([3, DTE1.shape[0]])
    DTE[0, :] = DTE1.reshape([DTE1.shape[0],])
    DTE[1, :] = DTE2.reshape([DTE2.shape[0],])
    DTE[2, :] = DTE3.reshape([DTE3.shape[0],])

    
    # Train a linear logistic regression model using different values of lambda
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

    
    # Train a linear logistic regression model using different values of lambda
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

"""



def k_fold_fusion3(D1, D2, D3, L, k, pi, Cfp, Cfn, pi_T, seed = 0):

    np.random.seed(seed)
    idx = np.random.permutation(D1.shape[1])

    start_index = 0
    elements = int(D1.shape[1] / k)

    llr = np.zeros([D1.shape[1], ])

    for count in range(k):

        if start_index + elements > D1.shape[1]:
            end_index = D1.shape[1]
        else:
            end_index = start_index + elements 

        # Define training and test partitions
        idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = np.zeros([3, idxTrain.shape[0]])  
        DTR[0, :] = D1[:, idxTrain].reshape([D1[:, idxTrain].shape[1], ])
        DTR[1, :] = D2[:, idxTrain].reshape([D2[:, idxTrain].shape[1], ])
        DTR[2, :] = D3[:, idxTrain].reshape([D3[:, idxTrain].shape[1], ])
        DTE = np.zeros([3, idxTest.shape[0]])
        DTE[0, :] = D1[:, idxTest].reshape([D1[:, idxTest].shape[1], ])
        DTE[1, :] = D2[:, idxTest].reshape([D2[:, idxTest].shape[1], ])
        DTE[2, :] = D3[:, idxTest].reshape([D3[:, idxTest].shape[1], ])

        LTR = L[idxTrain]
        LTE = L[idxTest]

        # Train a logistic regression model 
        llr[idxTest], _ = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, 0, pi_T, pi, Cfn, Cfp, calibration=False)
        
        start_index += elements

    # Calculate min and act DCF for the fusion
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)
    actDCF = act_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, actDCF





def k_fold_fusion2(D1, D2, L, k, pi, Cfp, Cfn, pi_T, seed = 0):

    np.random.seed(seed)
    idx = np.random.permutation(D1.shape[1])

    start_index = 0
    elements = int(D1.shape[1] / k)

    llr = np.zeros([D1.shape[1], ])

    for count in range(k):

        if start_index + elements > D1.shape[1]:
            end_index = D1.shape[1]
        else:
            end_index = start_index + elements 

        # Define training and test partitions
        idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = np.zeros([2, idxTrain.shape[0]])  
        DTR[0, :] = D1[:, idxTrain].reshape([D1[:, idxTrain].shape[1], ])
        DTR[1, :] = D2[:, idxTrain].reshape([D2[:, idxTrain].shape[1], ])
        DTE = np.zeros([2, idxTest.shape[0]])
        DTE[0, :] = D1[:, idxTest].reshape([D1[:, idxTest].shape[1], ])
        DTE[1, :] = D2[:, idxTest].reshape([D2[:, idxTest].shape[1], ])

        LTR = L[idxTrain]
        LTE = L[idxTest]

        # Train a logistic regression model 
        llr[idxTest], _ = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, 0, pi_T, pi, Cfn, Cfp, calibration=False)
        
        start_index += elements

    # Calculate min and act DCF for the fusion
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)
    actDCF = act_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, actDCF




def k_fold_calibration(D, L, k, pi, Cfp, Cfn, pi_T, seed=0):

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

    min_actDCF_cal = 1
    min_lambda = 0

    llr_cal = np.zeros([D.shape[1], ])
    opt_th_decisions = np.zeros([D.shape[1]])


    for count in range(k):

        if start_index + elements > D.shape[1]:
            end_index = D.shape[1]
        else:
            end_index = start_index + elements 

        # Define training and test partitions
        idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
    
        DTE = D[:, idxTest]
        LTE = L[idxTest]

        # Train a logistic regression model for score calibration
        llr_cal[idxTest], _ = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, 0.1, pi_T, pi, Cfn, Cfp, calibration=False)
        
        # Estimate optimal threshold on training set and perform decisions on test set
        _, opt_t = min_DCF(DTR.reshape([DTR.shape[1],]), pi, Cfn, Cfp, LTR)
        opt_th_decisions[idxTest] = 1 * (DTE.reshape([DTE.shape[1],]) > opt_t)

        start_index += elements

    # Subtract theoretical threshold to achieve calibrated scores
    llr_cal -= np.log(pi / (1 - pi))
    # Calculate act DCF for calibrated scores
    actDCF_cal = act_DCF(llr_cal, pi, Cfn, Cfp, L)

    # Calculate act DCF for optimal estimated threshold
    M = confusion_matrix(L, opt_th_decisions, 2)
    _, actDCF_estimated = Bayes_risk(M, pi, Cfn, Cfp)

    return actDCF_cal, actDCF_estimated, min_lambda


def analyse_scores_kfold(llr, pi, Cfn, Cfp, L, k, pi_T, name):
    
    actDCF = act_DCF(llr, pi, Cfn, Cfp, L)
    actDCF_cal, actDCF_estimated, min_lambda = k_fold_calibration(llr.reshape([1,llr.shape[0]]), L, k, pi, Cfp, Cfn, pi_T)

    print("\n\n******* "+name+" *********\n")
    print("act DCF: "+str(actDCF))
    print("act DCF, calibrated scores: "+ str(actDCF_cal) + " with min lambda: " + str(min_lambda))
    print("act DCF, estimated threshold: "+ str(actDCF_estimated))



def analyse_fusion_kfold2(D1, D2, L, k, pi, Cfp, Cfn, pi_T, name):
    minDCF, actDCF = k_fold_fusion2(D1.reshape([1,D1.shape[0]]), D2.reshape([1,D2.shape[0]]), L, k, pi, Cfp, Cfn, pi_T)

    print("\n\n******* " + name + " ********")
    print("min DCF: " + str(minDCF))
    print("act DCF: " + str(actDCF))



def analyse_fusion_kfold3(D1, D2, D3, L, k, pi, Cfp, Cfn, pi_T, name):
    minDCF, actDCF = k_fold_fusion3(D1.reshape([1,D1.shape[0]]), D2.reshape([1,D2.shape[0]]), D3.reshape([1,D3.shape[0]]), L, k, pi, Cfp, Cfn, pi_T)

    print("\n\n******* " + name + " ********")
    print("min DCF: " + str(minDCF))
    print("act DCF: " + str(actDCF))



if __name__ == "__main__":

    D, L = load("../Data/Train.txt")   
    DN = Z_score(D)
    k = 5
    pi = 0.5
    Cfp = 1
    Cfn = 1
    pi_T = 0.5
    (DNTR, LNTR), (DNTE, LNTE) = split_db_4to1(DN, L, seed = 0)


    ##################### SELECTED MODELS #######################
    # RBF-SVM, Z-normalized features, C=10, loggamma=-2, rebalancing
    # Quadratic logistic regression, Z-normalized features, lambda = 0
    # GMM, Z-normalized features, 8 components


    ###### RBF kernel SVM with C = 10 and log(gamma) = -2, rebalacing and Z-normalized features ######
    
    #_, llrSVM = SVM.k_fold_cross_validation(DN, L, SVM.kernel_SVM, k, pi, Cfp, Cfn, 10, pi_T, 1, rebalancing = True, type = "RBF", gamma = np.exp(-2), just_llr=True)
    #np.save("llrSVM.npy", llrSVM)
    llrSVM = np.load("../Data/llrSVM.npy")
    #analyze_scores(llrSVM, L, "RBF-SVM")
    analyse_scores_kfold(llrSVM, pi, Cfn, Cfp, L, k, pi_T, "SVM")

    ###### Quadratic LR, lambda = 0, Z-normalized features, no PCA ######

    #_, llrLR = LR.k_fold_cross_validation(DN, L, LR.quadratic_logistic_regression, k, pi, Cfp, Cfn, 0, pi_T)
    #np.save("llrLR.npy", llrLR)
    llrLR = np.load("../Data/llrLR.npy")
    #analyze_scores(llrLR, L, "Log Reg")
    analyse_scores_kfold(llrLR, pi, Cfn, Cfp, L, k, pi_T, "LR")
   
    ###### GMM, Z-normalized features, 8 components ######
    #_, llrGMM = GMM.k_fold_cross_validation(DN, L, k, pi, Cfp, Cfn, False, False, 8, seed = 0, just_llr = True)
    #np.save("llrGMM.npy", llrGMM)
    llrGMM = np.load("../Data/llrGMM.npy")
    #analyze_scores(llrGMM, L, "GMM")
    analyse_scores_kfold(llrGMM, pi, Cfn, Cfp, L, k, pi_T, "GMM")



    ###### Combined models ######
    #analyse_fusion_kfold2(llrSVM, llrLR, L, k, pi, Cfp, Cfn, pi_T, "SVM + LR")
    #analyse_fusion_kfold2(llrSVM, llrGMM, L, k, pi, Cfp, Cfn, pi_T, "SVM + GMM")
    # analyse_fusion_kfold3(llrSVM, llrLR, llrGMM, L, k, pi, Cfp, Cfn, pi_T, "SVM + LR + GMM")

    """
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
    """