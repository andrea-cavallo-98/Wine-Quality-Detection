
import numpy as np
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_4to1
from prediction_measurement import min_DCF, compute_llr
from data_visualization import Z_score
import matplotlib.pyplot as plt
from pca import compute_pca

def GAU_logpdf_ND(x, mu, C):
    M = x.shape[0]

    _, logSigma = np.linalg.slogdet(C)

    if x.shape[1] == 1:
        logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x-mu))
    else:
        logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.diagonal(np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x-mu)))

    return logN


def logpdf_GMM(X, gmm):
    S = np.zeros([len(gmm), X.shape[1]])

    for g in range(len(gmm)):
        S[g, :] = GAU_logpdf_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])

    # marginal log densities
    logdens = logsumexp(S, axis = 0)
    # posterior distributions
    logGamma = S - logdens
    gamma = np.exp(logGamma)
    return logdens, gamma


def GMM_EM_estimation(X, gmm, t, psi, diag = False, tied = False):

    curr_gmm = gmm
    ll = t + 1
    prev_ll = 0

    while abs(ll - prev_ll) >= t :
        # E-step: compute posterior probabilities
        logdens, gamma = logpdf_GMM(X, curr_gmm)
        if prev_ll == 0:
            prev_ll = sum(logdens) / X.shape[1]
        else:
            prev_ll = ll
        # M-step: update model parameters
        Z = np.sum(gamma, axis = 1)
        
        for g in range(len(gmm)):
            F = np.sum(gamma[g] * X, axis = 1)
            S = np.dot(gamma[g] * X, X.T)
            mu = (F / Z[g]).reshape([X.shape[0], 1])
            sigma = S / Z[g] - np.dot(mu, mu.T)
            if diag:
                sigma = sigma * np.eye(sigma.shape[0])
            U, s, _ = np.linalg.svd(sigma)
            # Add constraints on the covariance matrixes to avoid degenerate solutions
            s[s<psi] = psi
            covNew = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
            w = Z[g] / sum(Z)
            curr_gmm[g] = (w, mu, covNew)

        if tied:
            tot_sigma = np.zeros(curr_gmm[0][2].shape)
            for g in range(len(gmm)):
                tot_sigma += Z[g] * curr_gmm[g][2] 
            tot_sigma /= X.shape[1]
            U, s, _ = np.linalg.svd(tot_sigma)
            # Add constraints on the covariance matrixes to avoid degenerate solutions
            s[s<psi] = psi
            tot_sigma = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
            for g in range(len(gmm)):
                curr_gmm[g][2][:,:] = tot_sigma 

        logdens, _ = logpdf_GMM(X, curr_gmm)
        ll = sum(logdens) / X.shape[1]

    return curr_gmm, ll


def LBG(X, gmm, t, components, alpha, psi, diag, tied):

    for count in range(int(np.log2(components))):
        new_gmm = []
        for c in gmm:

            U, s, Vh = np.linalg.svd(c[2])
            d = U[:, 0:1] * s[0]**0.5 * alpha       

            new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) + d, c[2]))
            new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) - d, c[2]))

        gmm, ll = GMM_EM_estimation(X, new_gmm, t, psi, diag, tied)

    return gmm, ll


def GMM_classifier(DTR, LTR, DTE, LTE, n_classes, components, pi, Cfn, Cfp, diag, tied, t = 1e-6, psi = 0.01, alpha = 0.1):

    S = np.zeros([n_classes, DTE.shape[1]])
    for c in range(n_classes):
        covNew = np.cov(DTR[:, LTR == c])
        # Impose the constraint on the covariance matrix
        U, s, _ = np.linalg.svd(covNew)
        s[s<psi] = psi
        covNew = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
        # Start from max likelihood solution for one component
        starting_gmm = [(1.0, np.mean(DTR[:, LTR == c], axis = 1), covNew)]
        new_gmm, _ = LBG(DTR[:, LTR == c], starting_gmm, t, components, alpha, psi, diag, tied)
        logdens, _ = logpdf_GMM(DTE, new_gmm)
        S[c, :] = logdens
    
    llr = compute_llr(S)
    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, LTE)
    return llr, minDCF




def k_fold_cross_validation(D, L, k, pi, Cfp, Cfn, diag, tied, components, seed = 0):

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

        llr[idxTest], _ = GMM_classifier(DTR, LTR, DTE, LTE, 2, components, pi, Cfn, Cfp, diag, tied)
        start_index += elements

    minDCF, _ = min_DCF(llr, pi, Cfn, Cfp, L)

    return minDCF



if __name__ == "__main__":

    D, L = load("../Data/Train.txt")  
    (DTR, LTR), (DTE, LTE) = split_db_4to1(D, L)
    DN = Z_score(D)
    (DNTR, LNTR), (DNTE, LNTE) = split_db_4to1(DN, L)
    DG = np.load("gaussianized_features.npy")
    (DGTR, LGTR), (DGTE, LGTE) = split_db_4to1(DG, L)
    components_val = [2, 4, 8, 16]
    k = 5
    pi = 0.5
    Cfn = 1
    Cfp = 1
    fileName = "../Results/GMM_results.txt"

    DCF_z = np.zeros([4, len(components_val)])
    DCF_gaus = np.zeros([4, len(components_val)])
    
    with open(fileName, "w") as f:

        f.write("**** min DCF for different GMM models *****\n\n")

        for i, tied in enumerate([True, False]):
            f.write("\nTied: " + str(tied) + "\n")
            for j, diag in enumerate([False, True]):
                f.write("\nDiag: " + str(diag) + "\n")
                for t, components in enumerate(components_val):
                    f.write("\ncomponents: " + str(components) + "\n")
                    minDCF = k_fold_cross_validation(DN, L, k, pi, Cfp, Cfn, diag, tied, components)
                    DCF_z[2*i+j,t] = minDCF 
                    f.write("\nZ-norm: " + str(minDCF) + "\n")

                    minDCF = k_fold_cross_validation(DG, L, k, pi, Cfp, Cfn, diag, tied, components)
                    DCF_gaus[2*i+j,t] = minDCF 
                    f.write("\nGaussianized: " + str(minDCF) + "\n")

                    print("Finished tied: %s, diag: %s, components: %d" % (str(tied), str(diag), components))

    plt.figure()
    plt.plot(components_val, DCF_z[2,:])
    plt.plot(components_val, DCF_gaus[2,:])
    plt.xlabel("Components")
    plt.ylabel("min DCF")
    plt.legend(["Z-normalized", "Gaussianized"])
    plt.savefig("../Images/GMM_notied_nodiag")

    plt.figure()
    plt.plot(components_val, DCF_z[3,:])
    plt.plot(components_val, DCF_gaus[3,:])
    plt.xlabel("Components")
    plt.ylabel("min DCF")
    plt.legend(["Z-normalized", "Gaussianized"])
    plt.savefig("../Images/GMM_notied_diag")

    plt.figure()
    plt.plot(components_val, DCF_z[0,:])
    plt.plot(components_val, DCF_gaus[0,:])
    plt.xlabel("Components")
    plt.ylabel("min DCF")
    plt.legend(["Z-normalized", "Gaussianized"])
    plt.savefig("../Images/GMM_tied_nodiag")

    plt.figure()
    plt.plot(components_val, DCF_z[1,:])
    plt.plot(components_val, DCF_gaus[1,:])
    plt.xlabel("Components")
    plt.ylabel("min DCF")
    plt.legend(["Z-normalized", "Gaussianized"])
    plt.savefig("../Images/GMM_tied_diag")
