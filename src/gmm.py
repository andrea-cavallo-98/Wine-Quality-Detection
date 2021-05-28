
import numpy as np
from scipy.special import logsumexp
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_2to1


def GAU_logpdf_ND(x, mu, C):
    M = x.shape[0]
    if C.size != 1:
        _, logSigma = np.linalg.slogdet(C)
    else:
        logSigma = np.log(C)

    if C.size != 1:
        if x.shape[1] == 1:
            logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x-mu))
        else:
            logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.diagonal(np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x-mu)))
    else:
        if x.shape[1] == 1:
            logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.dot((x - mu).T /C, (x-mu))
        else:
            logN = -M/2*np.log(2*np.pi) - 0.5*logSigma - 0.5 * np.diagonal(np.dot((x - mu).T/C, (x-mu)))
 
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


def GMM_EM_estimation(X, gmm, t, psi):

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
            U, s, _ = np.linalg.svd(sigma)
            s[s<psi] = psi
            covNew = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
            w = Z[g] / sum(Z)
            curr_gmm[g] = (w, mu, covNew)

        logdens, _ = logpdf_GMM(X, curr_gmm)
        ll = sum(logdens) / X.shape[1]

    return curr_gmm, ll


def LBG(X, gmm, t, components, alpha, psi):

    for count in range(int(np.log2(components))):
        new_gmm = []
        for c in gmm:
            
            if c[2].size != 1:
                U, s, Vh = np.linalg.svd(c[2])
                d = U[:, 0:1] * s[0]**0.5 * alpha       
            else:
                d = alpha

            new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) + d, c[2]))
            new_gmm.append((c[0] / 2, c[1].reshape([X.shape[0], 1]) - d, c[2]))

        gmm, ll = GMM_EM_estimation(X, new_gmm, t, psi)

    return gmm, ll


def GMM_classifier(DTR, LTR, DTE, LTE, t, psi, alpha, n_classes, components):

    S = np.zeros([n_classes, DTE.shape[1]])
    for c in range(n_classes):
        covNew = np.cov(DTR[:, LTR == c])
        U, s, _ = np.linalg.svd(covNew)
        s[s<psi] = psi
        covNew = np.dot(U, s.reshape([s.shape[0], 1])*U.T)
        starting_gmm = [(1.0, np.mean(DTR[:, LTR == c], axis = 1), covNew)]
        new_gmm, _ = LBG(DTR[:, LTR == c], starting_gmm, t, components, alpha, psi)
        logdens, _ = logpdf_GMM(DTE, new_gmm)
        S[c, :] = logdens
    PredictedLabels = np.argmax(S, axis = 0)
    acc = sum(PredictedLabels == LTE) / LTE.shape[0]

    return 1-acc


if __name__ == "__main__":

    D, L = load("../Data/Train.txt")  
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    t = 1e-6
    psi = 0.01

    for components in [2, 4, 8, 16]:
        err = GMM_classifier(DTR, LTR, DTE, LTE, t, psi, 0.1, n_class, components)
        print("Components: %d, Error rate: %.4f" % (components, err))

