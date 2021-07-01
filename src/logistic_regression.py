import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import min_DCF, compute_llr
from data_visualization import Z_score
import matplotlib.pyplot as plt
from pca import compute_pca



def logreg_obj_wrap(DTR, LTR, l, pi_T):

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]

        Nt = sum(LTR == 1)
        Nf = sum(LTR == 0)

        J = l/2 * np.linalg.norm(w)**2 + pi_T / Nt * sum(np.log1p(np.exp( - (np.dot(w.T, DTR[:, LTR == 1]) + b )))) + \
            (1 - pi_T) / Nf * sum(np.log1p(np.exp((np.dot(w.T, DTR[:, LTR == 0]) + b ))))

        dJw = l * w - pi_T / Nt * np.sum(DTR[:, LTR == 1]/(1+np.exp(np.dot(w.T, DTR[:, LTR == 1]) + b)), axis = 1) + \
            (1-pi_T) / Nf * np.sum(DTR[:, LTR == 0]/(1+np.exp(-np.dot(w.T, DTR[:, LTR == 0]) - b)), axis = 1)

        dJb = -pi_T/Nt * np.sum(1/(1+np.exp(np.dot(w.T, DTR[:, LTR == 1]) + b))) + \
            (1-pi_T) / Nf * np.sum(1/(1+np.exp(-np.dot(w.T, DTR[:, LTR == 0]) - b)))

        dJ = np.concatenate((dJw, np.array(dJb).reshape(1,)))
        
        return J, dJ

    return logreg_obj


def linear_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp, calibration=False):
    
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, pi_T)

    optV, _, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad = False)   

    w, b = optV[0:-1], optV[-1]

    # Compute scores
    if calibration:
        s = np.dot(w.T, DTE) - b
    else:
        s = np.dot(w.T, DTE) + b

    minDCF, _ = min_DCF(s, pi, Cfn, Cfp, LTE)

    return s, minDCF


def map_to_feature_space(D):
    phi = np.zeros([D.shape[0]**2+D.shape[0], D.shape[1]])
    for index in range(D.shape[1]):
        x = D[:, index].reshape(D.shape[0], 1)
        # phi = [vec(x*x^T), x]^T
        phi[:, index] = np.concatenate((np.dot(x, x.T).reshape(x.shape[0]**2, 1), x)).reshape(phi.shape[0],)
    return phi

def quadratic_logistic_regression(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp):

    # Map training features to expanded feature space
    phi = map_to_feature_space(DTR)

    # Train a linear regression model on expanded feature space
    logreg_obj = logreg_obj_wrap(phi, LTR, l, pi_T)

    optV, _, _ = fmin_l_bfgs_b(logreg_obj, np.zeros(phi.shape[0] + 1), approx_grad = False)   

    w, b = optV[0:-1], optV[-1]

    # Map test features to expanded feature space
    phi_test = map_to_feature_space(DTE)

    # Compute scores
    s = np.dot(w.T, phi_test) + b

    minDCF, _ = min_DCF(s, pi, Cfn, Cfp, LTE)

    return s, minDCF


def k_fold_cross_validation(D, L, classifier, k, pi, Cfp, Cfn, l, pi_T, seed = 0, just_llr = False):

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

    if just_llr:
        minDCF = 0
    else:
        minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF, llr



if __name__ == "__main__":

    for LR_type in ["linear", "quadratic"]:
        D, L = load("../Data/Train.txt")    
        (DTR, LTR), (DTE, LTE) = split_db_4to1(D, L)
        DN = Z_score(D)
        (DNTR, LNTR), (DNTE, LNTE) = split_db_4to1(DN, L)
        DN10 = compute_pca(10, DN)
        (DNTR10, LNTR10), (DNTE10, LNTE10) = split_db_4to1(DN10, L)
        DG = np.load("../Data/gaussianized_features.npy")
        (DGTR, LGTR), (DGTE, LGTE) = split_db_4to1(DG, L)
        l_val = [0, 1e-6, 1e-4, 1e-2, 1, 100]
        pi_T = 0.5
        pi = 0.5
        Cfn = 1
        Cfp = 1
        k = 5

        img1 = "LR_lambda_kfold.png"
        img2 = "LR_lambda_single_split.png"
        fileName = "../Results/LR_results.txt"
        linear_or_quadratic = linear_logistic_regression
        if LR_type == "quadratic":
            fileName = "../Results/Quad_LR_results.txt"
            linear_or_quadratic = quadratic_logistic_regression
            img1 = "Quad_LR_lambda_kfold.png"
            img2 = "Quad_LR_lambda_single_split.png"

        with open(fileName, "w") as f:
            
            f.write("**** min DCF for different Logistic Regression models ****\n\n")
            
            f.write("Values of min DCF for values of lambda = [0, 1e-6, 1e-4, 1e-2, 1, 100]\n")
            f.write("\nRaw features\n")
            DCF_kfold_raw = []
            DCF_single_split_raw = []
            for l in l_val:
                minDCF, _ = k_fold_cross_validation(D, L, linear_or_quadratic, k, pi, Cfp, Cfn, l, pi_T, seed = 0)
                DCF_kfold_raw.append(minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = linear_or_quadratic(DTR, LTR, DTE, LTE, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_raw.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            f.write("\nZ-normalized features - no PCA\n")
            DCF_kfold_z = []
            DCF_single_split_z = []
            for l in l_val:
                minDCF, _ = k_fold_cross_validation(DN, L, linear_or_quadratic, k, pi, Cfp, Cfn, l, pi_T, seed = 0)
                DCF_kfold_z.append(minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = linear_or_quadratic(DNTR, LNTR, DNTE, LNTE, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_z.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            f.write("\nZ-normalized features - PCA = 10\n")
            DCF_kfold_z = []
            DCF_single_split_z = []
            for l in l_val:
                minDCF, _ = k_fold_cross_validation(DN10, L, linear_or_quadratic, k, pi, Cfp, Cfn, l, pi_T, seed = 0)
                DCF_kfold_z.append(minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = linear_or_quadratic(DNTR10, LNTR10, DNTE10, LNTE10, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_z.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            f.write("\nGaussianized features\n")
            DCF_kfold_gau = []
            DCF_single_split_gau = []
            for l in l_val:
                minDCF, _ = k_fold_cross_validation(DG, L, linear_or_quadratic, k, pi, Cfp, Cfn, l, pi_T, seed = 0)
                DCF_kfold_gau.append(minDCF)
                f.write("5-fold: " + str(minDCF))
                _, minDCF = linear_or_quadratic(DGTR, LGTR, DGTE, LGTE, l, pi_T, pi, Cfn, Cfp)
                DCF_single_split_gau.append(minDCF)
                f.write(" single split: " + str(minDCF) + "\n")
            
            
            plt.figure()
            plt.plot(l_val, DCF_kfold_raw)
            plt.plot(l_val, DCF_kfold_z)
            plt.plot(l_val, DCF_kfold_gau)
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend(["Raw", "Z-normalized", "Gaussianized"])
            plt.savefig("../Images/" + img1)

            plt.figure()
            plt.plot(l_val, DCF_single_split_raw)
            plt.plot(l_val, DCF_single_split_z)
            plt.plot(l_val, DCF_single_split_gau)
            plt.xscale("log")
            plt.xlabel(r"$\lambda$")
            plt.ylabel("min DCF")
            plt.legend(["Raw", "Z-normalized", "Gaussianized"])        
            plt.savefig("../Images/" + img2)
            