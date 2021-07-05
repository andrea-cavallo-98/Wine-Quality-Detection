"""

*** MODEL TUNING ***

Perform further analysis on the selected most promising models (RBF kernel SVM, quadratic logistic regression 
and GMM with 8 components). 

In particular, define functions to estimate the optimal threshold and apply it to a test set, to calibrate scores 
and to combine the outputs of different classifiers. All these actions are performed with k-fold cross validation. 

The main function trains the three selected models and analyses their actual DCF with no calibration, 
with score calibration and also by estimating an optimal threshold. Then, it also analyses the fusion of the 
selected models in terms of min DCF and actual DCF. Results are printed on the console.

"""

import numpy as np
from load_data import load
from prediction_measurement import act_DCF, min_DCF, confusion_matrix, Bayes_risk
import support_vector_machines as SVM
from data_visualization import Z_score
import logistic_regression as LR
import gmm as GMM

# Perform cross validation to evaluate the fusion of 3 models (scores are
# combined using linear logistic regression)
def k_fold_fusion3(D1, D2, D3, L, k, pi, Cfp, Cfn, pi_T, l, seed = 0):

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

        # Define training samples as arrays of the scores of the three different models
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
        llr[idxTest], _ = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
        
        start_index += elements

    # Calculate min and act DCF for the fusion
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)
    actDCF = act_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, actDCF


# Perform cross validation to evaluate the fusion of 2 models (scores are
# combined using linear logistic regression)
def k_fold_fusion2(D1, D2, L, k, pi, Cfp, Cfn, pi_T, l, seed = 0):

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

        # Define training samples as arrays of the scores of the three different models
        DTR = np.zeros([2, idxTrain.shape[0]])  
        DTR[0, :] = D1[:, idxTrain].reshape([D1[:, idxTrain].shape[1], ])
        DTR[1, :] = D2[:, idxTrain].reshape([D2[:, idxTrain].shape[1], ])
        DTE = np.zeros([2, idxTest.shape[0]])
        DTE[0, :] = D1[:, idxTest].reshape([D1[:, idxTest].shape[1], ])
        DTE[1, :] = D2[:, idxTest].reshape([D2[:, idxTest].shape[1], ])

        LTR = L[idxTrain]
        LTE = L[idxTest]

        # Train a logistic regression model 
        llr[idxTest], _ = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
        
        start_index += elements

    # Calculate min and act DCF for the fusion
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)
    actDCF = act_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, actDCF


# Perform cross validation to evaluate score calibration (scores are 
# calibrated with a linear logistic regression model)
def k_fold_calibration(D, L, k, pi, Cfp, Cfn, pi_T, l, seed=0, just_cal = False):

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

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
        llr_cal[idxTest], _ = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
        
        if not just_cal: # do not repeat optimal threshold estimation every time
            # Estimate optimal threshold on training set and perform decisions on test set
            _, opt_t = min_DCF(DTR.reshape([DTR.shape[1],]), pi, Cfn, Cfp, LTR)
            opt_th_decisions[idxTest] = 1 * (DTE.reshape([DTE.shape[1],]) > opt_t)

        start_index += elements

    # Subtract theoretical threshold to achieve calibrated scores
    llr_cal -= np.log(pi / (1 - pi))
    # Calculate act DCF for calibrated scores
    actDCF_cal = act_DCF(llr_cal, pi, Cfn, Cfp, L)

    if just_cal:
        actDCF_estimated = 0
    else:
        # Calculate act DCF for optimal estimated threshold
        M = confusion_matrix(L, opt_th_decisions, 2)
        _, actDCF_estimated = Bayes_risk(M, pi, Cfn, Cfp)

    return actDCF_cal, actDCF_estimated


# Perform cross validation to evaluate score calibration techniques and print results
def analyse_scores_kfold(llr, pi, Cfn, Cfp, L, k, pi_T, name):
    
    actDCF = act_DCF(llr, pi, Cfn, Cfp, L)
    # Choose the best value for lambda for logistic regression (try different ones)
    min_actDCF_cal = 1
    best_lambda = 0
    actDCF_estimated = 0
    for l in [0, 1e-6, 1e-3, 0.1, 1]:
        if l == 1: # last iteration, calculate also optimal estimated threshold
            actDCF_cal, actDCF_estimated = k_fold_calibration(llr.reshape([1,llr.shape[0]]), L, k, pi, Cfp, Cfn, pi_T, l, just_cal=False)
        else: # not the last iteration, just evaluate score calibration
            actDCF_cal, actDCF_estimated = k_fold_calibration(llr.reshape([1,llr.shape[0]]), L, k, pi, Cfp, Cfn, pi_T, l, just_cal=True)

        if actDCF_cal < min_actDCF_cal:
            min_actDCF_cal = actDCF_cal
            best_lambda = l

    print("\n\n******* "+name+" *********\n")
    print("act DCF: "+str(actDCF))
    print("act DCF, calibrated scores (logistic regression): "+ str(min_actDCF_cal) + " with best lambda: " + str(best_lambda))
    print("act DCF, estimated threshold: "+ str(actDCF_estimated))


# Perform cross validation to evaluate fusion of 2 models and print results
def analyse_fusion_kfold2(D1, D2, L, k, pi, Cfp, Cfn, pi_T, name):
    # Choose the best value for lambda for logistic regression (try different ones)
    min_minDCF = 1
    min_actDCF = 1
    best_lambda = 0
    for l in [0, 1e-6, 1e-3, 0.1, 1]:
        minDCF, actDCF = k_fold_fusion2(D1.reshape([1,D1.shape[0]]), D2.reshape([1,D2.shape[0]]), L, k, pi, Cfp, Cfn, pi_T, l)
        if minDCF < min_minDCF:
            min_minDCF = minDCF
            min_actDCF = actDCF
            best_lambda = l

    print("\n\n******* " + name + " ********")
    print("min DCF: " + str(min_minDCF))
    print("act DCF: " + str(min_actDCF))
    print("best lambda: " + str(best_lambda))


# Perform cross validation to evaluate fusion of 3 models and print results
def analyse_fusion_kfold3(D1, D2, D3, L, k, pi, Cfp, Cfn, pi_T, name):
    # Choose the best value for lambda for logistic regression (try different ones)
    min_minDCF = 1
    min_actDCF = 1
    best_lambda = 0
    for l in [0, 1e-6, 1e-3, 0.1, 1]:
        minDCF, actDCF = k_fold_fusion3(D1.reshape([1,D1.shape[0]]), D2.reshape([1,D2.shape[0]]), D3.reshape([1,D3.shape[0]]), L, k, pi, Cfp, Cfn, pi_T, l)
        if minDCF < min_minDCF:
            min_minDCF = minDCF
            min_actDCF = actDCF
            best_lambda = l

    print("\n\n******* " + name + " ********")
    print("min DCF: " + str(min_minDCF))
    print("act DCF: " + str(min_actDCF))
    print("best lambda: " + str(best_lambda))




if __name__ == "__main__":

    ### Analyze score calibration and model fusion for the selected models

    D, L = load("../Data/Train.txt")   
    DN = Z_score(D)
    k = 5
    pi = 0.5
    Cfp = 1
    Cfn = 1
    pi_T = 0.5


    ##################### SELECTED MODELS #######################
    # RBF-SVM, Z-normalized features, C=10, loggamma=-2, rebalancing
    # Quadratic logistic regression, Z-normalized features, lambda = 0
    # GMM, Z-normalized features, 8 components


    ###### RBF kernel SVM with C = 10 and log(gamma) = -2, rebalacing and Z-normalized features ######
    
    _, llrSVM = SVM.k_fold_cross_validation(DN, L, SVM.kernel_SVM, k, pi, Cfp, Cfn, 10, pi_T, 1, rebalancing = True, type = "RBF", gamma = np.exp(-2), just_llr=True)
    np.save("llrSVM.npy", llrSVM)
    llrSVM = np.load("../Data/llrSVM.npy")
    analyse_scores_kfold(llrSVM, pi, Cfn, Cfp, L, k, pi_T, "SVM")

    ###### Quadratic LR, lambda = 0, Z-normalized features, no PCA ######

    _, llrLR = LR.k_fold_cross_validation(DN, L, LR.quadratic_logistic_regression, k, pi, Cfp, Cfn, 0, pi_T)
    np.save("llrLR.npy", llrLR)
    llrLR = np.load("../Data/llrLR.npy")
    analyse_scores_kfold(llrLR, pi, Cfn, Cfp, L, k, pi_T, "LR")
   
    ###### GMM, Z-normalized features, 8 components ######
    _, llrGMM = GMM.k_fold_cross_validation(DN, L, k, pi, Cfp, Cfn, False, False, 8, seed = 0, just_llr = True)
    np.save("llrGMM.npy", llrGMM)
    llrGMM = np.load("../Data/llrGMM.npy")
    analyse_scores_kfold(llrGMM, pi, Cfn, Cfp, L, k, pi_T, "GMM")



    ###### Combined models ######
    analyse_fusion_kfold2(llrSVM, llrLR, L, k, pi, Cfp, Cfn, pi_T, "SVM + LR")
    analyse_fusion_kfold2(llrSVM, llrGMM, L, k, pi, Cfp, Cfn, pi_T, "SVM + GMM")
    analyse_fusion_kfold3(llrSVM, llrLR, llrGMM, L, k, pi, Cfp, Cfn, pi_T, "SVM + LR + GMM")
