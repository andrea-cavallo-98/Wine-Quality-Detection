
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import min_DCF
from data_visualization import Z_score
import matplotlib.pyplot as plt
from pca import compute_pca


def obj_function_gradient_wrapper(H_hat):
    def obj_function_gradient(alpha):
        obj_function = 1/2 * np.dot(np.dot(alpha.T, H_hat), alpha) - sum(alpha)
        gradient = np.dot(H_hat, alpha) - 1
        return obj_function, gradient.reshape([gradient.size, ])

    return obj_function_gradient


def kernel(x1, x2, type, d = 0, c = 0, gamma = 0, csi = 0):
    if type == "poly":
        return (np.dot(x1.T, x2) + c) ** d + csi

    else: 
        if type == "RBF":
            k = np.zeros([x1.shape[1], x2.shape[1]])
            for index1 in range(x1.shape[1]):
                for index2 in range(x2.shape[1]):
                    k[index1, index2] = np.exp(-gamma * ((x1[:, index1] - x2[:, index2]) * (x1[:, index1] - x2[:, index2])).sum()) + csi
            return k

        else:
            return 0


def linear_SVM(DTR, LTR, DTE, LTE, C, K, pi, Cfp, Cfn, pi_T, rebalancing = True):
    
    D_hat = np.concatenate((DTR, K * np.array(np.ones([1, DTR.shape[1]]))))
    
    # Compute H_hat
    Z = np.ones(LTR.shape)
    Z[LTR == 0] = -1
    ZiZj = np.dot(Z.reshape([Z.shape[0], 1]), Z.reshape([Z.shape[0], 1]).T)
    H_hat = ZiZj * np.dot(D_hat.T, D_hat)

    # Optimize the objective function
    obj_function_gradient = obj_function_gradient_wrapper(H_hat)
    B = np.zeros([DTR.shape[1], 2])
    if rebalancing:
        pi_T_emp = sum(LTR == 1) / DTR.shape[1]
        Ct = C * pi_T / pi_T_emp
        Cf = C * (1 - pi_T) / (1 - pi_T_emp)
        B[LTR == 1, 1] = Ct
        B[LTR == 0, 1] = Cf
    else:
        B[:, 1] = C
    optAlpha, _, _ = fmin_l_bfgs_b(obj_function_gradient, np.zeros(DTR.shape[1]),
                                        approx_grad = False, bounds = B, factr = 10000.0)

    # Recover primal solution
    w_hat = np.sum(optAlpha * Z * D_hat, axis = 1)

    # Compute extended data matrix for test set
    T_hat = np.concatenate((DTE, K * np.array(np.ones([1, DTE.shape[1]]))))
    
    # Compute scores, predictions and accuracy
    S = np.dot(w_hat.T, T_hat)
    
    minDCF, _ = min_DCF(S, pi, Cfn, Cfp, LTE)

    return S, minDCF


def kernel_SVM(DTR, LTR, DTE, LTE, C, type, pi, Cfn, Cfp, pi_T, d = 0, c = 0, gamma = 0, csi = 0, rebalancing = True, store_model = False):
        
    # Compute H_hat
    Z = np.ones(LTR.shape)
    Z[LTR == 0] = -1
    ZiZj = np.dot(Z.reshape([Z.shape[0], 1]), Z.reshape([Z.shape[0], 1]).T)
    H_hat = ZiZj * kernel(DTR, DTR, type, d, c, gamma, csi)

    # Optimize the objective function
    obj_function_gradient = obj_function_gradient_wrapper(H_hat)
    B = np.zeros([DTR.shape[1], 2])
    if rebalancing:
        pi_T_emp = sum(LTR == 1) / DTR.shape[1]
        Ct = C * pi_T / pi_T_emp
        Cf = C * (1 - pi_T) / (1 - pi_T_emp)
        B[LTR == 1, 1] = Ct
        B[LTR == 0, 1] = Cf
    else:
        B[:, 1] = C
    optAlpha, _, _ = fmin_l_bfgs_b(obj_function_gradient, np.zeros(DTR.shape[1]),
                                        approx_grad = False, bounds = B, factr = 10000.0)

    if store_model:
        np.save("SVM_alpha.npy", optAlpha)

    # Compute scores
    S = np.sum((optAlpha * Z).reshape([DTR.shape[1], 1]) * kernel(DTR, DTE, type, d, c, gamma, csi), axis = 0)

    minDCF, _ = min_DCF(S, pi, Cfn, Cfp, LTE)

    return S, minDCF


def k_fold_cross_validation(D, L, classifier, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = True, 
                            gamma = 0, seed = 0, type = "", just_llr = False, store_model = False):

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

        if type == "": # linear SVM
            llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing)
        else: # kernel SVM
            llr[idxTest], _ = classifier(DTR, LTR, DTE, LTE, C, type, pi, Cfn, Cfp, pi_T, gamma=gamma,
                                     rebalancing=rebalancing, d = 2, csi=K_SVM**0.5, c = 1, store_model=store_model)

        start_index += elements

    if just_llr:
        minDCF = 0
    else:
        minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, llr




if __name__ == "__main__":

    D, L = load("../Data/Train.txt")    
    (DTR, LTR), (DTE, LTE) = split_db_4to1(D, L)
    DN = Z_score(D)
    (DNTR, LNTR), (DNTE, LNTE) = split_db_4to1(DN, L)
    DG = np.load("../Data/gaussianized_features.npy")
    (DGTR, LGTR), (DGTE, LGTE) = split_db_4to1(DG, L)
    C_val = [1e-1, 1, 10]
    pi_T = 0.5
    pi = 0.5
    Cfn = 1
    Cfp = 1
    k = 5
    K_SVM = 1

    """
    LINEAR SVM
    """
    """
    img1_val = ["SVM_C_kfold_nobal.png", "SVM_C_kfold_bal.png"]
    img2_val = ["SVM_C_single_split_nobal.png", "SVM_C_single_split_bal.png"]
    fileName = "../Results/linear_SVM_results.txt"
    linear_or_quadratic = linear_SVM
    doRebalancing = True
    with open(fileName, "w") as f:
        
        f.write("**** min DCF for different linear SVM models ****\n\n")
        f.write("Values of min DCF for values of C = [0, 1e-1, 1, 10]\n")

        for i, doRebalancing in enumerate([False, True]):

            f.write("\n Rebalancing: " + str(doRebalancing) + "\n")

            f.write("\nRaw features\n")
            DCF_kfold_raw = []
            DCF_single_split_raw = []
            for C in C_val:
                minDCF, _ = k_fold_cross_validation(D, L, linear_or_quadratic, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = doRebalancing, seed = 0)
                DCF_kfold_raw.append(minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = linear_or_quadratic(DTR, LTR, DTE, LTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing = doRebalancing)
                DCF_single_split_raw.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished raw features")

            f.write("\nZ-normalized features - no PCA\n")
            DCF_kfold_z = []
            DCF_single_split_z = []
            for C in C_val:
                minDCF, _ = k_fold_cross_validation(DN, L, linear_or_quadratic, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = doRebalancing, seed = 0)
                DCF_kfold_z.append(minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = linear_or_quadratic(DNTR, LNTR, DNTE, LNTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing = doRebalancing)
                DCF_single_split_z.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Z-normalized features")

            f.write("\nGaussianized features\n")
            DCF_kfold_gau = []
            DCF_single_split_gau = []
            for C in C_val:
                minDCF, _ = k_fold_cross_validation(DG, L, linear_or_quadratic, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = doRebalancing, seed = 0)
                DCF_kfold_gau.append(minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = linear_or_quadratic(DGTR, LGTR, DGTE, LGTE, C, K_SVM, pi, Cfp, Cfn, pi_T, rebalancing = doRebalancing)
                DCF_single_split_gau.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Gaussianized features")

            img1 = img1_val[i]
            img2 = img2_val[i]

            plt.figure()
            plt.plot(C_val, DCF_kfold_raw)
            plt.plot(C_val, DCF_kfold_z)
            plt.plot(C_val, DCF_kfold_gau)
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend(["Raw", "Z-normalized", "Gaussianized"])
            plt.savefig("../Images/" + img1)

            plt.figure()
            plt.plot(C_val, DCF_single_split_raw)
            plt.plot(C_val, DCF_single_split_z)
            plt.plot(C_val, DCF_single_split_gau)
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend(["Raw", "Z-normalized", "Gaussianized"])        
            plt.savefig("../Images/" + img2)

    """
    """
    QUADRATIC KERNEL SVM
    """
    """
    fileName = "../Results/quad_SVM_results.txt"
    with open(fileName, "w") as f:
        
        f.write("**** min DCF for different quadratic kernel SVM models ****\n\n")
        f.write("Values of min DCF for values of C = [1e-1, 1, 10]\n")

        f.write("\nZ-normalized features - no PCA - no rebalancing\n")
        DCF_kfold_z_nobal = []
        DCF_single_split_z_nobal = []
        for C in C_val:
            minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = False, type = "poly")
            DCF_kfold_z_nobal.append(minDCF)
            f.write("5-fold: " + str(minDCF))
            _, minDCF = kernel_SVM(DNTR, LNTR, DNTE, LNTE, C, "poly", pi, Cfn, Cfp, pi_T, d = 2, csi = K_SVM**0.5, rebalancing = False, c=1)
            DCF_single_split_z_nobal.append(minDCF)
            f.write(" single split: " + str(minDCF) + "\n")
        
        print("Finished Z-normalized features - no rebalancing")

        f.write("\nZ-normalized features - no PCA - rebalancing\n")
        DCF_kfold_z_bal = []
        DCF_single_split_z_bal = []
        for C in C_val:
            minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = True, type = "poly")
            DCF_kfold_z_bal.append(minDCF)
            f.write("5-fold: " + str(minDCF))
            _, minDCF = kernel_SVM(DNTR, LNTR, DNTE, LNTE, C, "poly", pi, Cfn, Cfp, pi_T, d = 2, csi = K_SVM**0.5, rebalancing = True)
            DCF_single_split_z_bal.append(minDCF)
            f.write(" single split: " + str(minDCF) + "\n")
        
        print("Finished Z-normalized features - rebalancing")

        img1 = "quad_SVM_C_kfold.png"
        img2 = "quad_SVM_C_single_split.png"

        plt.figure()
        plt.plot(C_val, DCF_kfold_z_nobal, marker='o', linestyle='dashed', color="red")
        plt.plot(C_val, DCF_kfold_z_bal, marker='o', linestyle='dashed', color="blue")
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("min DCF")
        plt.legend(["No balancing", "Balancing"])
        plt.savefig("../Images/" + img1)

        plt.figure()
        plt.plot(C_val, DCF_single_split_z_nobal, marker='o', linestyle='dashed', color="red")
        plt.plot(C_val, DCF_single_split_z_bal, marker='o', linestyle='dashed', color="blue")
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("min DCF")
        plt.legend(["No balancing", "Balancing"])
        plt.savefig("../Images/" + img2)
    """
    """
    RBF KERNEL SVM
    """
    """
    fileName = "../Results/RBF_SVM_results.txt"
    gamma_val = [np.exp(-1), np.exp(-2)]

    with open(fileName, "w") as f:

        DCF_kfold_z_nobal = np.zeros([len(gamma_val), len(C_val)])
        DCF_kfold_z_bal = np.zeros([len(gamma_val), len(C_val)])
        DCF_single_split_z_nobal = np.zeros([len(gamma_val), len(C_val)])
        DCF_single_split_z_bal = np.zeros([len(gamma_val), len(C_val)])

        for i, gamma in enumerate(gamma_val):
            f.write("**** min DCF for different quadratic kernel SVM models ****\n\n")
            f.write("Values of min DCF for values of C = [1e-1, 1, 10]\n")

            f.write("\nZ-normalized features - no PCA - no rebalancing\n")
            for j,C in enumerate(C_val):
                minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = False, type = "RBF", gamma = gamma)
                DCF_kfold_z_nobal[i, j] = (minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = kernel_SVM(DNTR, LNTR, DNTE, LNTE, C, "RBF", pi, Cfn, Cfp, pi_T, gamma = gamma, csi = K_SVM**0.5, rebalancing = False)
                DCF_single_split_z_nobal[i,j] = (minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Z-normalized features - no rebalancing")

            f.write("\nZ-normalized features - no PCA - rebalancing\n")
            for C in C_val:
                minDCF, _ = k_fold_cross_validation(DN, L, kernel_SVM, k, pi, Cfp, Cfn, C, pi_T, K_SVM, rebalancing = True, type = "RBF", gamma = gamma)
                DCF_kfold_z_bal[i,j] = minDCF
                f.write("5-fold: " + str(minDCF))
                _, minDCF = kernel_SVM(DNTR, LNTR, DNTE, LNTE, C, "RBF", pi, Cfn, Cfp, pi_T, gamma = gamma, csi = K_SVM**0.5, rebalancing = True)
                DCF_single_split_z_bal[i,j] = minDCF
                f.write(" single split: " + str(minDCF) + "\n")
            
            print("Finished Z-normalized features - rebalancing")

        img1 = "RBF_SVM_C_kfold_bal.png"
        img2 = "RBF_SVM_C_single_split_bal.png"

        plt.figure()
        plt.plot(C_val, DCF_kfold_z_nobal[0,:], marker='o', linestyle='dashed', color="red")
        plt.plot(C_val, DCF_kfold_z_bal[0,:], marker='o', linestyle='dashed', color="blue")
        plt.plot(C_val, DCF_kfold_z_nobal[1,:], marker='o', linestyle='dashed', color="green")
        plt.plot(C_val, DCF_kfold_z_bal[1,:], marker='o', linestyle='dashed', color="black")
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("min DCF")
        plt.legend([r"$log \gamma = -1$"+", No balancing", r"$log \gamma = -1$"+", Balancing",
            r"$log \gamma = -2$"+", No balancing", r"$log \gamma = -2$"+", Balancing"])
        plt.savefig("../Images/" + img1)

        plt.figure()
        plt.plot(C_val, DCF_single_split_z_nobal[0,:], marker='o', linestyle='dashed', color="red")
        plt.plot(C_val, DCF_single_split_z_bal[0,:], marker='o', linestyle='dashed', color="blue")
        plt.plot(C_val, DCF_single_split_z_nobal[1,:], marker='o', linestyle='dashed', color="green")
        plt.plot(C_val, DCF_single_split_z_bal[1,:], marker='o', linestyle='dashed', color="black")
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("min DCF")
        plt.legend([r"$log \gamma = -1$"+", No balancing", r"$log \gamma = -1$"+", Balancing",
            r"$log \gamma = -2$"+", No balancing", r"$log \gamma = -2$"+", Balancing"])
        plt.savefig("../Images/" + img2)
        """