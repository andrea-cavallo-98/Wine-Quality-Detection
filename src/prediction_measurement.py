import numpy as np

# Compute confusion matrix with given predictions
def confusion_matrix(true_labels, predicted_labels, n_labels):

    conf_matrix = np.zeros([n_labels, n_labels])

    for current_true_label in range(n_labels):
        for current_predicted_label in range(n_labels):
            conf_matrix[current_predicted_label, current_true_label] = sum(predicted_labels[true_labels == current_true_label] == current_predicted_label)

    return conf_matrix

# Take optimal Bayes decisions (using optimal theoretical threshold)
def optimal_Bayes_decisions(llr, pi, Cfn, Cfp):

    t = - np.log(pi * Cfn / ((1 - pi) * Cfp))

    PredictedLabels = np.zeros([llr.shape[0]])
    PredictedLabels[llr > t] = 1

    return PredictedLabels

# Calculate unnormalized DCF and normalized DCF
def Bayes_risk(M, pi, Cfn, Cfp):
    
    FNR = M[0,1] / (M[0,1] + M[1,1])
    FPR = M[1,0] / (M[0,0] + M[1,0])

    DCFu = pi * Cfn * FNR + (1 - pi) * Cfp * FPR
    B_dummy = min(pi * Cfn, (1 - pi) * Cfp)
    DCF = DCFu / B_dummy

    return DCFu, DCF

# Calculate actual DCF using optimal Bayes decisions
def act_DCF(llr, pi, Cfn, Cfp, true_labels):
    opt_decisions = optimal_Bayes_decisions(llr, pi, Cfn, Cfp)
    M = confusion_matrix(true_labels, opt_decisions, 2)
    _, DCF = Bayes_risk(M, pi, Cfn, Cfp)
    return DCF

# Calculate min DCF by trying all possible thresholds
def min_DCF(llr, pi, Cfn, Cfp, true_labels):

    possible_t = np.concatenate((np.array([min(llr) - 0.1]), (np.unique(llr)), np.array([max(llr) + 0.1])))

    minDCF = 10
    opt_t = 0

    for t in possible_t:
        PredictedLabels = np.zeros([llr.shape[0]])
        PredictedLabels[llr > t] = 1
        M = confusion_matrix(true_labels, PredictedLabels, 2)
        _, DCF = Bayes_risk(M, pi, Cfn, Cfp)
        if DCF < minDCF:
            minDCF = DCF
            opt_t = t

    return minDCF, opt_t

# Calculate data to print the ROC plot (FPR and TPR for different thresholds)
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

    return FPR, TPR

# Calculate data to print Bayes error plots (minDCF and DCF for different applications)
def Bayes_error_plots(llr, true_labels):
    effPriorLogOdds = np.linspace(-3,3,21)
    DCF = np.zeros([effPriorLogOdds.shape[0]])
    minDCF = np.zeros([effPriorLogOdds.shape[0]])

    for index, p_tilde in enumerate(effPriorLogOdds):
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        pred = optimal_Bayes_decisions(llr, pi_tilde, 1, 1)
        M = confusion_matrix(true_labels, pred, 2)
        _, DCF[index] = Bayes_risk(M, pi_tilde, 1, 1)
        minDCF[index], _ = min_DCF(llr, pi_tilde, 1, 1, true_labels)

    return DCF, minDCF

# Compute llr from class conditional log probabilities (just subtract the
# values for the two classes)
def compute_llr(s):
    if s.shape[0] != 2:
        return 0
    return s[1, :] - s[0, :]



