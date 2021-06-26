import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import act_DCF, min_DCF, confusion_matrix, Bayes_risk
import support_vector_machines as SVM
from data_visualization import Z_score, Z_score_eval
import logistic_regression as LR
import gmm as GMM
from pca import compute_pca, compute_pca_eval
import gaussian_models as GM
import logistic_regression as LR


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

    return minDCF, actDCF


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

    return minDCF, actDCF



if __name__ == "__main__":

    # Training and test data, raw features
    DTR, LTR = load("../Data/Train.txt")
    DTE, LTE = load("../Data/Test.txt")  
    # Z-score normalization
    DNTR = Z_score(DTR)
    DNTE = Z_score_eval(DTR, DTE)
    # Gaussianization
    DGTR = np.load("gaussianized_features.npy")
    DGTE = np.load("gaus_test.npy")
    # 10-PCA, Gaussianized features
    DGTR10 = compute_pca(10, DGTR)
    DGTE10 = compute_pca_eval(10, DGTR, DGTE)
    # 9-PCA, Gaussianized features
    DGTR9 = compute_pca(9, DGTR)
    DGTE9 = compute_pca_eval(9, DGTR, DGTE)   
    # 10-PCA, Z-normalized features 
    DNTR10 = compute_pca(10, DNTR)
    DNTE10 = compute_pca_eval(10, DNTR, DNTE)

    use_logs = True
    pi = 0.5
    pi_T = 0.5
    Cfn = 1
    Cfp = 1
    K_SVM = 1
    C_val = [1e-1, 1, 10]

    """

    #######################################
    # Gaussian models
    #######################################

    fileName = "../Results/gaussian_results_eval.txt"

    with open(fileName, "w") as f:
        
        f.write("*** min DCF for different gaussian models ***\n\n")
        f.write("Gaussianized features - no PCA\n")

        _, DCF_MVG = GM.MVG(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = GM.naive_Bayes(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = GM.tied_MVG(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DGTR, LTR, DGTE, LTE, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
            " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))

        f.write("\n\nGaussianized features - PCA = 10\n")

        _, DCF_MVG = GM.MVG(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = GM.naive_Bayes(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = GM.tied_MVG(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DGTR10, LTR, DGTE10, LTE, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) 
            + " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        
        f.write("\n\nGaussianized features - PCA = 9\n")
    
        _, DCF_MVG = GM.MVG(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = GM.naive_Bayes(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = GM.tied_MVG(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DGTR9, LTR, DGTE9, LTE, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) 
            + " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))
        
        f.write("\n\nRaw features - no PCA\n")
       
        _, DCF_MVG = GM.MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_naive_Bayes = GM.naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_MVG = GM.tied_MVG(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)
        _, DCF_tied_naive_Bayes = GM.tied_naive_Bayes(DTR, LTR, DTE, LTE, pi, Cfp, Cfn, use_logs)

        f.write("MVG: " + str(DCF_MVG) + " naive Bayes: " + str(DCF_naive_Bayes) + 
                    " tied MVG: " + str(DCF_tied_MVG) + " tied naive Bayes: " + str(DCF_tied_naive_Bayes))

    """
    """
    ################################################
    # Logistic regression
    ################################################

    l_val = [0, 1e-6, 1e-4, 1e-2, 1, 100]

    for LR_type in ["linear", "quadratic"]:

        fileName = "../Results/LR_results_evalPCA.txt"
        linear_or_quadratic = LR.linear_logistic_regression
        if LR_type == "quadratic":
            fileName = "../Results/Quad_LR_results_evalPCA.txt"
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
    """

    """
    ################################
    # Linear SVM
    ################################

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

    """
    """
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
            _, minDCF = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, C, "poly", pi, Cfn, Cfp, pi_T, d = 2, csi = K_SVM**0.5, rebalancing = True)
            f.write(" single split: " + str(minDCF) + "\n")
        
        print("Finished Z-normalized features - rebalancing")
    """
    """
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
    """
    """
    #######################
    # GMM
    #######################

    components = 16
    fileName = "../Results/GMM_results_eval.txt"

    with open(fileName, "w") as f:

        f.write("**** min DCF for different GMM models *****\n\n")

        for tied in [True, False]:
            f.write("\nTied: " + str(tied) + "\n")
            for diag in [False, True]:
                f.write("\nDiag: " + str(diag) + "\n")
                
                _, minDCF = GMM.GMM_classifier(DNTR, LTR, DNTE, LTE, 2, components, pi, Cfn, Cfp, diag, tied, f=f, type="Z-norm")
                _, minDCF = GMM.GMM_classifier(DGTR, LTR, DGTE, LTE, 2, components, pi, Cfn, Cfp, diag, tied, f=f, type="Gaussianized")

                print("Finished tied: %s, diag: %s" % (str(tied), str(diag)))
    """

    ########################
    # Fusions of best models
    ########################

    ##################### SELECTED MODELS #######################
    # RBF-SVM, Z-normalized features, C=10, loggamma=-2, rebalancing
    # Quadratic logistic regression, Z-normalized features, lambda = 0
    # GMM, Z-normalized features, 8 components

    # Load scores on training data
    llrSVMTrain = np.load("llrSVM.npy")
    llrLRTrain = np.load("llrLR.npy")
    llrGMMTrain = np.load("llrGMM.npy")

    # Train models on all training dataset and get scores on evaluation dataset
    # llrSVMTest, _ = SVM.kernel_SVM(DNTR, LTR, DNTE, LTE, 10, "RBF", pi, Cfn, Cfp, pi_T, gamma = np.exp(-2), csi = K_SVM**0.5, rebalancing = True)
    # llrLRTest, _ = LR.quadratic_logistic_regression(DNTR, LTR, DNTE, LTE, 0, pi_T, pi, Cfn, Cfp)
    # llrGMMTest, _ = GMM.GMM_classifier(DNTR, LTR, DNTE, LTE, 2, 8, pi, Cfn, Cfp, False, False)
    # np.save("llrSVMTest.npy", llrSVMTest)
    # np.save("llrLRTest.npy", llrLRTest)
    # np.save("llrGMMTest.npy", llrGMMTest)

    llrSVMTest = np.load("llrSVMTest.npy")
    llrLRTest = np.load("llrLRTest.npy")
    llrGMMTest = np.load("llrGMMTest.npy")

    # Evaluate performance of combined models 
    # (LR model trained on training set scores and evaluated on test set scores)

    fileName = "../Results/fusions_results_eval.txt"
    with open(fileName, "w") as f:

        f.write("*********** Actual DCF of single models ************")
        actDCF = act_DCF(llrSVMTest, pi, Cfn, Cfp, LTE)
        f.write("\n\nSVM: " + str(actDCF))
        actDCF = act_DCF(llrLRTest, pi, Cfn, Cfp, LTE)
        f.write("\nLR: " + str(actDCF))
        actDCF = act_DCF(llrGMMTest, pi, Cfn, Cfp, LTE)
        f.write("\nGMM: " + str(actDCF))

        f.write("\n\n*********** SVM + LR ************\n\n")
        minDCF, actDCF = evaluate_model_fusion(llrSVMTrain, llrLRTrain, llrSVMTest, llrLRTest, LTR, LTE)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))
        
        f.write("\n\n*********** SVM + GMM ************\n\n")
        minDCF, actDCF = evaluate_model_fusion(llrSVMTrain, llrGMMTrain, llrSVMTest, llrGMMTest, LTR, LTE)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))

        f.write("\n\n*********** SVM + LR + GMM ************\n\n")
        minDCF, actDCF = evaluate_model_fusion3(llrSVMTrain, llrLRTrain, llrGMMTrain, llrSVMTest, llrLRTest, llrGMMTest, LTR, LTE)
        f.write("min DCF: " + str(minDCF) + " act DCF: " + str(actDCF))
