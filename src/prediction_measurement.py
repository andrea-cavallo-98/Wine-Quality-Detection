
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1

def confusion_matrix(true_labels, predicted_labels, n_labels):

    conf_matrix = np.zeros([n_labels, n_labels])

    for current_true_label in range(n_labels):
        for current_predicted_label in range(n_labels):
            conf_matrix[current_predicted_label, current_true_label] = sum(predicted_labels[true_labels == current_true_label] == current_predicted_label)

    return conf_matrix


def optimal_Bayes_decisions(llr, pi, Cfn, Cfp):

    t = - np.log(pi * Cfn / ((1 - pi) * Cfp))

    PredictedLabels = np.zeros([llr.shape[0]])
    PredictedLabels[llr > t] = 1

    return PredictedLabels


def Bayes_risk(M, pi, Cfn, Cfp):
    
    FNR = M[0,1] / (M[0,1] + M[1,1])
    FPR = M[1,0] / (M[0,0] + M[1,0])

    DCFu = pi * Cfn * FNR + (1 - pi) * Cfp * FPR
    B_dummy = min(pi * Cfn, (1 - pi) * Cfp)
    DCF = DCFu / B_dummy

    return DCFu, DCF


def min_DCF(llr, pi, Cfn, Cfp, true_labels):

    possible_t = np.concatenate((np.array([min(llr) - 0.1]), (np.unique(llr)), np.array([max(llr) + 0.1])))

    minDCF = 10

    for t in possible_t:
        PredictedLabels = np.zeros([llr.shape[0]])
        PredictedLabels[llr > t] = 1
        M = confusion_matrix(true_labels, PredictedLabels, 2)
        _, DCF = Bayes_risk(M, pi, Cfn, Cfp)
        if DCF < minDCF:
            minDCF = DCF

    return minDCF


def ROC_curve(llr, true_labels):

    possible_t = np.concatenate((np.array([min(llr) - 0.1]), (np.unique(llr)), np.array([max(llr) + 0.1])))
    FPR = np.zeros([possible_t.shape[0]])
    TPR = np.zeros([possible_t.shape[0]])

    for index, t in enumerate(possible_t):

        PredictedLabels = np.zeros([llr.shape[0]])
        PredictedLabels[llr > t] = 1
        M = confusion_matrix(true_labels, PredictedLabels, 2)
        FNR = M[0,1] / (M[0,1] + M[1,1])
        FPR[index] = M[1,0] / (M[0,0] + M[1,0])
        TPR[index] = 1 - FNR

    plt.figure()
    plt.plot(FPR, TPR)
    plt.grid(True)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


def Bayes_error_plots(llr, true_labels):
    effPriorLogOdds = np.linspace(-3,3,21)
    DCF = np.zeros([effPriorLogOdds.shape[0]])
    minDCF = np.zeros([effPriorLogOdds.shape[0]])

    for index, p_tilde in enumerate(effPriorLogOdds):
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        pred = optimal_Bayes_decisions(llr, pi_tilde, 1, 1)
        M = confusion_matrix(true_labels, pred, 2)
        _, DCF[index] = Bayes_risk(M, pi_tilde, 1, 1)
        minDCF[index] = min_DCF(llr, pi_tilde, 1, 1, true_labels)

    plt.figure()
    plt.plot(effPriorLogOdds, DCF, label="DCF", color="r")
    plt.plot(effPriorLogOdds, minDCF, label="min DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.show()

def compute_llr(s):
    if s.shape[0] != 2:
        return 0
    return s[1, :] - s[0, :]



if __name__ == "__main__":

    ### Try to compute some confusion matrixes from previous labs

    # LTE, MVG_labels, tied_covariance_labels = compute_predictions_IRIS()
    # Commedia_labels, Commedia_predicted_labels = compute_predictions_Commedia()

    # c1 = confusion_matrix(LTE, MVG_labels, 3)
    # c2 = confusion_matrix(LTE, tied_covariance_labels, 3)
    # c3 = confusion_matrix(Commedia_labels, Commedia_predicted_labels, 3)

    # print("\nConfusion matrix with MVG classifier on IRIS dataset: ")
    # print(c1)
    # print("\nConfusion matrix with tied covariances classifier on IRIS dataset: ")
    # print(c2)
    # print("\nConfusion matrix on Divina Commedia: ")
    # print(c3)

    ## Compute optimal Bayes decisions for gaussian models

    D, L = load("../Data/Train.txt")  
    (DTR, LTR), (DTE, LTE) = split_db_4to1(D, L)
  
    """
    _, _, _, S_MVG = MVG(DTR, LTR, DTE, LTE, True)
    _, _, _, S_naive_Bayes = naive_Bayes(DTR, LTR, DTE, LTE, True)
    _, _, _, S_tied_covariance = tied_MVG(DTR, LTR, DTE, LTE, True)
    llr_MVG = compute_llr(S_MVG)
    llr_naive_Bayes = compute_llr(S_naive_Bayes)
    llr_tied_covariance = compute_llr(S_tied_covariance)

    ## For different models, compare DCF and min DCF on the Inferno vs Paradiso binary task

    pi = [0.5, 0.8, 0.5, 0.8]
    Cfn = [1, 1, 10, 1]
    Cfp = [1, 1, 1, 10]

    for llr in [llr_MVG, llr_naive_Bayes, llr_tied_covariance]:

        for index in range(4):
            pred = optimal_Bayes_decisions(llr, pi[index], Cfn[index], Cfp[index])
            c = confusion_matrix(LTE, pred, 2)
            print(sum(pred == LTE) / LTE.shape[0])
            #print(c)
            DCFu, DCF = Bayes_risk(c, pi[index], Cfn[index], Cfp[index])
            minDCF = min_DCF(llr, pi[index], Cfn[index], Cfp[index], LTE)
            print("%.3f | %.3f" % (DCF, minDCF))

    """
        # Print ROC curve 
        #ROC_curve(llr, LTE)

        # Print Bayes error plots 
        #Bayes_error_plots(llr, LTE)
