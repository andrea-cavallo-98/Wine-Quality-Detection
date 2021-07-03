import numpy as np
import matplotlib.pyplot as plt
from load_data import load, split_db_4to1
from prediction_measurement import Bayes_error_plots, act_DCF, min_DCF, confusion_matrix, Bayes_risk, ROC_curve
import support_vector_machines as SVM
from data_visualization import Z_score, Z_score_eval
import logistic_regression as LR
import gmm as GMM
from pca import compute_pca, compute_pca_eval
import gaussian_models as GM
import logistic_regression as LR

# Evaluate score calibration by training a logistic regression model on training data
# and evaluating it on test data
def evaluate_score_calibration(llrTrain, llrTest, LTR, LTE, pi, Cfn, Cfp):
    
    # Train score calibration on training llr and evaluate results on test llr
    s, _ = LR.linear_logistic_regression(llrTrain.reshape([1,llrTrain.shape[0]]), LTR, 
        llrTest.reshape([1,llrTest.shape[0]]), LTE, 0, 0.5, pi, Cfn, Cfp)

    # Subtract theoretical threshold to achieve calibrated scores
    s -= np.log(pi/(1-pi))
    actDCF_cal = act_DCF(s, pi, Cfn, Cfp, LTE)

    # Select optimal threshold on training llr and evaluate results on test llr
    _, opt_t = min_DCF(llrTrain, pi, Cfn, Cfp, LTR)
    PredictedLabels = np.zeros(LTE.shape)
    PredictedLabels[llrTest > opt_t] = 1
    M = confusion_matrix(LTE, PredictedLabels, 2)
    _, actDCF_estimated = Bayes_risk(M, pi, Cfn, Cfp)

    return actDCF_cal, actDCF_estimated

# Evaluate fusion of 2 models by training a logistic regression model on training data
# and evaluating it on test data
def evaluate_model_fusion(llrTrain1, llrTrain2, llrTest1, llrTest2, LTR, LTE):
    
    # Define DTR and DTE as combinations of llr from different models
    DTR = np.zeros([2, llrTrain1.shape[0]])
    DTR[0, :] = llrTrain1.reshape([llrTrain1.shape[0],])
    DTR[1, :] = llrTrain2.reshape([llrTrain2.shape[0],])
    DTE = np.zeros([2, llrTest1.shape[0]])
    DTE[0, :] = llrTest1.reshape([llrTest1.shape[0],])
    DTE[1, :] = llrTest2.reshape([llrTest2.shape[0],])

    s, minDCF = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, 0, 0.5, pi, Cfn, Cfp, calibration=False)
    actDCF = act_DCF(s, pi, Cfn, Cfp, LTE)

    return minDCF, actDCF, s

# Evaluate fusion of 3 models by training a logistic regression model on training data
# and evaluating it on test data
def evaluate_model_fusion3(llrTrain1, llrTrain2, llrTrain3, llrTest1, llrTest2, llrTest3, LTR, LTE):
    
    # Define DTR and DTE as combinations of llr from different models
    DTR = np.zeros([3, llrTrain1.shape[0]])
    DTR[0, :] = llrTrain1.reshape([llrTrain1.shape[0],])
    DTR[1, :] = llrTrain2.reshape([llrTrain2.shape[0],])
    DTR[2, :] = llrTrain3.reshape([llrTrain3.shape[0],])
    DTE = np.zeros([3, llrTest1.shape[0]])
    DTE[0, :] = llrTest1.reshape([llrTest1.shape[0],])
    DTE[1, :] = llrTest2.reshape([llrTest2.shape[0],])
    DTE[2, :] = llrTest3.reshape([llrTest3.shape[0],])

    s, minDCF = LR.linear_logistic_regression(DTR, LTR, DTE, LTE, 0, 0.5, pi, Cfn, Cfp, calibration=False)
    actDCF = act_DCF(s, pi, Cfn, Cfp, LTE)

    return minDCF, actDCF, s


def plot_ROC_curve(FPR1, TPR1, FPR2, TPR2, FPR3, TPR3, l1, l2, l3, figName):
    plt.figure()
    plt.plot(FPR1, TPR1, 'b')
    plt.plot(FPR2, TPR2, 'r')
    plt.plot(FPR3, TPR3, 'g')
    plt.grid(True)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend([l1, l2, l3])
    plt.savefig("../Images/" + figName +".png")   


def plot_Bayes_error(D1, D2, D3, D4, D5, D6, l1, l2, l3, l4, l5, l6, figName):

    effPriorLogOdds = np.linspace(-3,3,21)

    plt.figure()
    plt.plot(effPriorLogOdds, D1, label=l1, color='red')
    plt.plot(effPriorLogOdds, D2, label=l2, color='red', linestyle='dashed')
    plt.plot(effPriorLogOdds, D3, label=l3, color="b")
    plt.plot(effPriorLogOdds, D4, label=l4, color='b', linestyle='dashed')
    plt.plot(effPriorLogOdds, D5, label=l5, color="g")
    plt.plot(effPriorLogOdds, D6, label=l6, color="g", linestyle='dashed')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.savefig("../Images/" + figName +".png")   



def plot_Bayes_error4(D1, D2, D3, D4, l1, l2, l3, l4, figName):

    effPriorLogOdds = np.linspace(-3,3,21)

    plt.figure()
    plt.plot(effPriorLogOdds, D1, label=l1, color='red')
    plt.plot(effPriorLogOdds, D2, label=l2, color='g')
    plt.plot(effPriorLogOdds, D3, label=l3, color="b")
    plt.plot(effPriorLogOdds, D4, label=l4, color='b', linestyle='dashed')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.savefig("../Images/" + figName +".png")   


if __name__ == "__main__":

    ### Test the performances of different models on the test dataset

    # Training and test data, raw features
    DTR, LTR = load("../Data/Train.txt")
    DTE, LTE = load("../Data/Test.txt")  
    # Z-score normalization
    DNTR = Z_score(DTR)
    DNTE = Z_score_eval(DTR, DTE)
    # Gaussianization
    DGTR = np.load("../Data/gaussianized_features.npy")
    DGTE = np.load("../Data/gaus_test.npy")
    # 10-PCA, Gaussianized features
    DGTR10 = compute_pca(10, DGTR)
    DGTE10 = compute_pca_eval(10, DGTR, DGTE)
    # 9-PCA, Gaussianized features
    DGTR9 = compute_pca(9, DGTR)
    DGTE9 = compute_pca_eval(9, DGTR, DGTE)   
    # 10-PCA, Z-normalized features 
    DNTR10 = compute_pca(10, DNTR)
    DNTE10 = compute_pca_eval(10, DNTR, DNTE)

    pi = 0.5
    pi_T = 0.5
    Cfn = 1
    Cfp = 1
    K_SVM = 1
    C_val = [1e-1, 1, 10]
    use_80_data = False
    
    if use_80_data:
        # Train only using 80% of data
        (DTR, _), (_, _) = split_db_4to1(DTR, LTR)
        (DNTR, _), (_, _) = split_db_4to1(DNTR, LTR)
        (DNTR10, _), (_, _) = split_db_4to1(DNTR10, LTR)
        (DGTR, _), (_, _) = split_db_4to1(DGTR, LTR)
        (DGTR10, _), (_, _) = split_db_4to1(DGTR10, LTR)
        (DGTR9, LTR), (_, _) = split_db_4to1(DGTR9, LTR)
    
    #######################################
    # Gaussian models
    #######################################

    if use_80_data:
        fileName = "../Results/gaussian_results_eval80.txt"
    else:
        fileName = "../Results/gaussian_results_eval.txt"

    with open(fileName, "w") as f:
        
        f.write("*** min DCF for different gaussian models ***\n\n")
        f.write("Gaussianized features - no PCA\n")

        _, DCF_MVG = GM.MVG(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn)
        _, DCF_naive_Bayes = GM.naive_Bayes(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn)
        _, DCF_tied_MVG = GM.tied_MVG(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn)
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn )

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
            " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))

        f.write("\n\nGaussianized features - PCA = 10\n")

        _, DCF_MVG = GM.MVG(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn )
        _, DCF_naive_Bayes = GM.naive_Bayes(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn )
        _, DCF_tied_MVG = GM.tied_MVG(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn )
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn )

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) 
            + " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        
        f.write("\n\nGaussianized features - PCA = 9\n")
    
        _, DCF_MVG = GM.MVG(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn )
        _, DCF_naive_Bayes = GM.naive_Bayes(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn )
        _, DCF_tied_MVG = GM.tied_MVG(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn )
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn )

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) 
            + " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        
        f.write("\n\nRaw features - no PCA\n")
       
        _, DCF_MVG = GM.MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn )
        _, DCF_naive_Bayes = GM.naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn )
        _, DCF_tied_MVG = GM.tied_MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn )
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn )

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
                    " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
    
    ################################################
    # Logistic regression
    ################################################

    l_val = [0, 1e-6, 1e-4, 1e-2, 1, 100]

    for LR_type in ["linear", "quadratic"]:

        if use_80_data:
            fileName = "../Results/LR_results_eval80.txt"
            linear_or_quadratic = LR.linear_logistic_regression
            if LR_type == "quadratic":
                fileName = "../Results/Quad_LR_results_eval80.txt"
                linear_or_quadratic = LR.quadratic_logistic_regression
        else:
            fileName = "../Results/LR_results_eval.txt"
            linear_or_quadratic = LR.linear_logistic_regression
            if LR_type == "quadratic":
                fileName = "../Results/Quad_LR_results_eval.txt"
                linear_or_quadratic = LR.quadratic_logistic_regression


        with open(fileName, "w") as f:
            
            f.write("**** min DCF for different Logistic Regression models ****\n\n")
            
            f.write("Values of min DCF for values of lambda = [0, 1e-6, 1e-4, 1e-2, 1, 100]\n")
            f.write("\nRaw features\n")
            DCF_single_split_raw = []
            for l in l_val:
                _, minDCF = linear_or_quadratic(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_raw.append(minDCF)
                f.write("min DCF: " + str(minDCF) + "\n")
            
            f.write("\nZ-normalized features - no PCA\n")
            DCF_single_split_z = []
            for l in l_val:
                _, minDCF = linear_or_quadratic(DNTR, LTR, DNTE, LTE, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_z.append(minDCF)
                f.write("min DCF: " + str(minDCF) + "\n")
            
            f.write("\nZ-normalized features - PCA = 10\n")
            DCF_single_split_z = []
            for l in l_val:
                _, minDCF = linear_or_quadratic(DNTR10, LTR, DNTE10, LTE, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_z.append(minDCF)
                f.write("min DCF: " + str(minDCF) + "\n")
            
            f.write("\nGaussianized features\n")
            DCF_single_split_gau = []
            for l in l_val:
                _, minDCF = linear_or_quadratic(DGTR, LTR, DGTE, LTE, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_gau.append(minDCF)
                f.write("min DCF: " + str(minDCF) + "\n")
    

    ################################
    # Linear SVM
    ################################

    if use_80_data:
        fileName = "../Results/linear_SVM_results_eval80.txt"
    else:
        fileName = "../Results/linear_SVM_results_eval.txt"
    linear_or_quadratic = SVM.linear_SVM
    doRebalancing = True
    with open(fileName, "w") as f:
        
        f.write("**** min DCF for different linear SVM models ****\n\n")
        f.write("Values of min DCF for values of C = [0, 1e-1, 1, 10]\n")

        for i, doRebalancing in enumerate([False, True]):

            f.write("\n Rebalancing: " + str(doRebalancing) + "\n")

            f.write("\nRaw features\n")
            DCF_single_split_raw = []
            for C in C_val:
                _, minDCF = linear_or_quadratic(DTR, LTR, DTE, LTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing = doRebalancing)
                DCF_single_split_raw.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished raw features")

            f.write("\nZ-normalized features - no PCA\n")
            DCF_single_split_z = []
            for C in C_val:
                _, minDCF = linear_or_quadratic(DNTR, LTR, DNTE, LTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing = doRebalancing)
                DCF_single_split_z.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Z-normalized features")

            f.write("\nGaussianized features\n")
            DCF_single_split_gau = []
            for C in C_val:
                _, minDCF = linear_or_quadratic(DGTR, LTR, DGTE, LTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing = doRebalancing)
                DCF_single_split_gau.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Gaussianized features")

    
    
    ###############################
    # Quadratic kernel SVM
    ###############################
    
    fileName = "../Results/quad_SVM_results_eval.txt"
    with open(fileName, "w") as f:
        
        f.write("**** min DCF for different quadratic kernel SVM models ****\n\n")
        f.write("Values of min DCF for values of C = [1e-1, 1, 10]\n")

        f.write("\nZ-normalized features - no PCA - no rebalancing\n")
        for C in C_val:
            _, minDCF = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, C, "poly", pi, Cfn, Cfp, pi_T, d = 2, csi = K_SVM**0.5, rebalancing = False, c=1)
            f.write(" single split: " + str(minDCF) + "\n")
        
        print("Finished Z-normalized features - no rebalancing")

        f.write("\nZ-normalized features - no PCA - rebalancing\n")
        for C in C_val:
            _, minDCF = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, C, "poly", pi, Cfn, Cfp, pi_T, d = 2, csi = K_SVM**0.5, rebalancing = True, c=1)
            f.write(" single split: " + str(minDCF) + "\n")
        
        print("Finished Z-normalized features - rebalancing")
    
    
    ###########################
    # RBF kernel SVM
    ###########################
    
    fileName = "../Results/RBF_SVM_results_eval.txt"
    gamma_val = [np.exp(-1), np.exp(-2)]

    with open(fileName, "w") as f:

        for i, gamma in enumerate(gamma_val):
            f.write("**** min DCF for different quadratic kernel SVM models ****\n\n")
            f.write("Values of min DCF for values of C = [1e-1, 1, 10]\n")

            f.write("\nZ-normalized features - no PCA - no rebalancing\n")
            for j,C in enumerate(C_val):
                _, minDCF = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, C, "RBF", pi, Cfn, Cfp, pi_T, gamma = gamma, csi = K_SVM**0.5, rebalancing = False)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Z-normalized features - no rebalancing")

            f.write("\nZ-normalized features - no PCA - rebalancing\n")
            for C in C_val:
                _, minDCF = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, C, "RBF", pi, Cfn, Cfp, pi_T, gamma = gamma, csi = K_SVM**0.5, rebalancing = True)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Z-normalized features - rebalancing")
    
    
    #######################
    # GMM
    #######################

    components = 16
    fileName = "../Results/GMM_results_evalTied.txt"

    with open(fileName, "w") as f:

        f.write("**** min DCF for different GMM models *****\n\n")

        for tied in [True]:
            f.write("\nTied: " + str(tied) + "\n")
            for diag in [False, True]:
                f.write("\nDiag: " + str(diag) + "\n")
                
                _, minDCF = GMM.GMM_classifier(DNTR, LTR, DNTE, LTE, 2, components, pi, Cfn, Cfp, diag, tied, f=f, type="Z-norm")
                _, minDCF = GMM.GMM_classifier(DGTR, LTR, DGTE, LTE, 2, components, pi, Cfn, Cfp, diag, tied, f=f, type="Gaussianized")

                print("Finished tied: %s, diag: %s" % (str(tied), str(diag)))

    
    # Load scores on training data
    llrSVMTrain = np.load("../Data/llrSVM.npy")
    llrLRTrain = np.load("../Data/llrLR.npy")
    llrGMMTrain = np.load("../Data/llrGMM.npy")
    llrSVMTest = np.load("../Data/llrSVMTest.npy")
    llrLRTest = np.load("../Data/llrLRTest.npy")
    llrGMMTest = np.load("../Data/llrGMMTest.npy")

    actDCF_cal, actDCF_estimated = evaluate_score_calibration(llrSVMTrain, llrSVMTest, LTR, LTE, pi, Cfn, Cfp)

    print("Calibrated: " + str(actDCF_cal))
    print("Estimated: " + str(actDCF_estimated))

    
    ########################
    # Fusions of best models
    ########################

    ##################### SELECTED MODELS #######################
    # RBF-SVM, Z-normalized features, C=10, loggamma=-2, rebalancing
    # Quadratic logistic regression, Z-normalized features, lambda = 0
    # GMM, Z-normalized features, 8 components

    # Load scores on training data
    llrSVMTrain = np.load("../Data/llrSVM.npy")
    llrLRTrain = np.load("../Data/llrLR.npy")
    llrGMMTrain = np.load("../Data/llrGMM.npy")

    # Train models on all training dataset and get scores on evaluation dataset
    # llrSVMTest, _ = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, 10, "RBF", pi, Cfn, Cfp, pi_T, gamma = np.exp(-2), csi = K_SVM**0.5, rebalancing = True)
    # llrLRTest, _ = LR.quadratic_logistic_regression(DNTR, LTR, DNTE, LTE, 0, pi_T, pi, Cfn, Cfp)
    # llrGMMTest, _ = GMM.GMM_classifier(DNTR, LTR, DNTE, LTE, 2, 8, pi, Cfn, Cfp, False, False)
    # np.save("llrSVMTest.npy", llrSVMTest)
    # np.save("llrLRTest.npy", llrLRTest)
    # np.save("llrGMMTest.npy", llrGMMTest)

    llrSVMTest = np.load("../Data/llrSVMTest.npy")
    llrLRTest = np.load("../Data/llrLRTest.npy")
    llrGMMTest = np.load("../Data/llrGMMTest.npy")

    # Evaluate performance of combined models 
    # (LR model trained on training set scores and evaluated on test set scores)
    
    fileName = "../Results/fusions_results_eval2.txt"
    with open(fileName, "w") as f:

        f.write("*********** Actual DCF of single models ************")
        actDCF = act_DCF(llrSVMTest, pi, Cfn, Cfp, LTE)
        actDCF_cal, actDCF_estimated = evaluate_score_calibration(llrSVMTrain, llrSVMTest, LTR, LTE, pi, Cfn, Cfp)
        f.write("\n\nSVM: actual: " + str(actDCF) + " calibrated: " + str(actDCF_cal) + " estimated: " + str(actDCF_estimated))
        actDCF = act_DCF(llrLRTest, pi, Cfn, Cfp, LTE)
        actDCF_cal, actDCF_estimated = evaluate_score_calibration(llrLRTrain, llrLRTest, LTR, LTE, pi, Cfn, Cfp)
        f.write("\nLR: actual: " + str(actDCF) + " calibrated: " + str(actDCF_cal) + " estimated: " + str(actDCF_estimated))
        actDCF = act_DCF(llrGMMTest, pi, Cfn, Cfp, LTE)
        actDCF_cal, actDCF_estimated = evaluate_score_calibration(llrGMMTrain, llrGMMTest, LTR, LTE, pi, Cfn, Cfp)
        f.write("\nGMM: actual: " + str(actDCF) + " calibrated: " + str(actDCF_cal) + " estimated: " + str(actDCF_estimated))

        f.write("\n\n*********** SVM + LR ************\n\n")
        minDCF, actDCF, _ = evaluate_model_fusion(llrSVMTrain, llrLRTrain, llrSVMTest, llrLRTest, LTR, LTE)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))
        
        f.write("\n\n*********** SVM + GMM ************\n\n")
        minDCF, actDCF, _ = evaluate_model_fusion(llrSVMTrain, llrGMMTrain, llrSVMTest, llrGMMTest, LTR, LTE)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))

        f.write("\n\n*********** SVM + LR + GMM ************\n\n")
        minDCF, actDCF, _ = evaluate_model_fusion3(llrSVMTrain, llrLRTrain, llrGMMTrain, llrSVMTest, llrLRTest, llrGMMTest, LTR, LTE)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))
    
    
    
    ### ROC plots

    FPR_SVM, TPR_SVM = ROC_curve(llrSVMTest, LTE)
    FPR_LR, TPR_LR = ROC_curve(llrLRTest, LTE)
    FPR_GMM, TPR_GMM = ROC_curve(llrGMMTest, LTE)

    _, _, llrSVMLR = evaluate_model_fusion(llrSVMTrain, llrLRTrain, llrSVMTest, llrLRTest, LTR, LTE)
    _, _, llrSVMGMM = evaluate_model_fusion(llrSVMTrain, llrGMMTrain, llrSVMTest, llrGMMTest, LTR, LTE)
    _, _, llrSVMLRGMM = evaluate_model_fusion3(llrSVMTrain, llrLRTrain, llrGMMTrain, llrSVMTest, llrLRTest, llrGMMTest, LTR, LTE)

    FPR_SVMLR, TPR_SVMLR = ROC_curve(llrSVMLR, LTE)
    FPR_SVMGMM, TPR_SVMGMM = ROC_curve(llrSVMGMM, LTE)
    FPR_SVMLRGMM, TPR_SVMLRGMM = ROC_curve(llrSVMLRGMM, LTE)

    np.save("../Data/FPR_SVM.npy", FPR_SVM)
    np.save("../Data/TPR_SVM.npy", TPR_SVM)
    np.save("../Data/FPR_LR.npy", FPR_LR)
    np.save("../Data/TPR_LR.npy", TPR_LR)
    np.save("../Data/FPR_GMM.npy", FPR_GMM)
    np.save("../Data/TPR_GMM.npy", TPR_GMM)
    np.save("../Data/FPR_SVMLR.npy", FPR_SVMLR)
    np.save("../Data/TPR_SVMLR.npy", TPR_SVMLR)
    np.save("../Data/FPR_SVMGMM.npy", FPR_SVMGMM)
    np.save("../Data/TPR_SVMGMM.npy", TPR_SVMGMM)
    np.save("../Data/FPR_SVMLRGMM.npy", FPR_SVMLRGMM)
    np.save("../Data/TPR_SVMLRGMM.npy", TPR_SVMLRGMM)
    
    FPR_SVM = np.load("../Data/FPR_SVM.npy")
    TPR_SVM = np.load("../Data/TPR_SVM.npy")
    FPR_LR = np.load("../Data/FPR_LR.npy")
    TPR_LR = np.load("../Data/TPR_LR.npy")
    FPR_GMM = np.load("../Data/FPR_GMM.npy")
    TPR_GMM = np.load("../Data/TPR_GMM.npy")
    FPR_SVMLR = np.load("../Data/FPR_SVMLR.npy")
    TPR_SVMLR = np.load("../Data/TPR_SVMLR.npy")
    FPR_SVMGMM = np.load("../Data/FPR_SVMGMM.npy")
    TPR_SVMGMM = np.load("../Data/TPR_SVMGMM.npy")
    FPR_SVMLRGMM = np.load("../Data/FPR_SVMLRGMM.npy")
    TPR_SVMLRGMM = np.load("../Data/TPR_SVMLRGMM.npy")

    plot_ROC_curve(FPR_SVM, TPR_SVM, FPR_LR, TPR_LR, FPR_GMM, TPR_GMM, "SVM", "LR", "GMM", "ROC_eval1")
    plot_ROC_curve(FPR_SVMLR, TPR_SVMLR, FPR_SVMGMM, TPR_SVMGMM, FPR_SVMLRGMM, TPR_SVMLRGMM, "SVM+LR", "SVM+GMM", "SVM+LR+GMM", "ROC_eval2")
    plot_ROC_curve(FPR_SVM, TPR_SVM, FPR_LR, TPR_LR, FPR_SVMLRGMM, TPR_SVMLRGMM, "SVM", "LR", "SVM+LR+GMM", "ROC_eval3")
    
    
    ### min DCF plots

    DCF_SVM, minDCF_SVM = Bayes_error_plots(llrSVMTest, LTE)
    DCF_LR, minDCF_LR = Bayes_error_plots(llrLRTest, LTE)
    DCF_GMM, minDCF_GMM = Bayes_error_plots(llrGMMTest, LTE)

    _, _, llrSVMLR = evaluate_model_fusion(llrSVMTrain, llrLRTrain, llrSVMTest, llrLRTest, LTR, LTE)
    _, _, llrSVMGMM = evaluate_model_fusion(llrSVMTrain, llrGMMTrain, llrSVMTest, llrGMMTest, LTR, LTE)
    _, _, llrSVMLRGMM = evaluate_model_fusion3(llrSVMTrain, llrLRTrain, llrGMMTrain, llrSVMTest, llrLRTest, llrGMMTest, LTR, LTE)

    DCF_SVMLR, minDCF_SVMLR = Bayes_error_plots(llrSVMLR, LTE)
    DCF_SVMGMM, minDCF_SVMGMM = Bayes_error_plots(llrSVMGMM, LTE)
    DCF_SVMLRGMM, minDCF_SVMLRGMM = Bayes_error_plots(llrSVMLRGMM, LTE)

    np.save("../Data/DCF_SVM.npy", DCF_SVM)
    np.save("../Data/minDCF_SVM.npy", minDCF_SVM)
    np.save("../Data/DCF_LR.npy", DCF_LR)
    np.save("../Data/minDCF_LR.npy", minDCF_LR)
    np.save("../Data/DCF_GMM.npy", DCF_GMM)
    np.save("../Data/minDCF_GMM.npy", minDCF_GMM)
    np.save("../Data/DCF_SVMLR.npy", DCF_SVMLR)
    np.save("../Data/minDCF_SVMLR.npy", minDCF_SVMLR)
    np.save("../Data/DCF_SVMGMM.npy", DCF_SVMGMM)
    np.save("../Data/minDCF_SVMGMM.npy", minDCF_SVMGMM)
    np.save("../Data/DCF_SVMLRGMM.npy", DCF_SVMLRGMM)
    np.save("../Data/minDCF_SVMLRGMM.npy", minDCF_SVMLRGMM)

    
    DCF_SVM = np.load("../Data/DCF_SVM.npy")
    minDCF_SVM = np.load("../Data/minDCF_SVM.npy")
    DCF_LR = np.load("../Data/DCF_LR.npy")
    minDCF_LR = np.load("../Data/minDCF_LR.npy")
    DCF_GMM = np.load("../Data/DCF_GMM.npy")
    minDCF_GMM = np.load("../Data/minDCF_GMM.npy")
    DCF_SVMLR = np.load("../Data/DCF_SVMLR.npy")
    minDCF_SVMLR = np.load("../Data/minDCF_SVMLR.npy")
    DCF_SVMGMM = np.load("../Data/DCF_SVMGMM.npy")
    minDCF_SVMGMM = np.load("../Data/minDCF_SVMGMM.npy")
    DCF_SVMLRGMM = np.load("../Data/DCF_SVMLRGMM.npy")
    minDCF_SVMLRGMM = np.load("../Data/minDCF_SVMLRGMM.npy")


    plot_Bayes_error(DCF_SVM, minDCF_SVM, DCF_LR, minDCF_LR, DCF_GMM, minDCF_GMM, 
        "SVM: act DCF", "SVM: min DCF", "LR: act DCF", "LR: min DCF", "GMM: act DCF", "GMM: min DCF", "Bayes_error1_eval")
    plot_Bayes_error(DCF_SVMLR, minDCF_SVMLR, DCF_SVMGMM, minDCF_SVMGMM, DCF_SVMLRGMM, minDCF_SVMLRGMM, 
        "SVM+LR: act DCF", "SVM+LR: min DCF", "SVM+GMM: act DCF", "SVM+GMM: min DCF", "SVM+LR+GMM: act DCF", "SVM+LR+GMM: min DCF", "Bayes_error2_eval")
    plot_Bayes_error4(DCF_SVM, DCF_LR, DCF_SVMLRGMM, minDCF_SVMLRGMM, 
        "SVM: act DCF", "LR: act DCF", "SVM+LR+GMM: act DCF", "SVM+LR+GMM: min DCF", "Bayes_error3_eval")
    