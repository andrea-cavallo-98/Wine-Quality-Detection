
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from load_data import load, attributes_names, class_names, n_attr, n_class, split_db_2to1


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


def linear_SVM(DTR, LTR, DTE, LTE, C, K):
    
    D_hat = np.concatenate((DTR, K * np.array(np.ones([1, DTR.shape[1]]))))
    
    # Compute H_hat

    Z = np.ones(LTR.shape)
    Z[LTR == 0] = -1
    ZiZj = np.dot(Z.reshape([Z.shape[0], 1]), Z.reshape([Z.shape[0], 1]).T)
    H_hat = ZiZj * np.dot(D_hat.T, D_hat)

    # Optimize the objective function
    obj_function_gradient = obj_function_gradient_wrapper(H_hat)
    B = np.zeros([DTR.shape[1], 2])
    B[:, 1] = C
    optAlpha, dual_obj, _ = fmin_l_bfgs_b(obj_function_gradient, np.zeros(DTR.shape[1]),
                                        approx_grad = False, bounds = B, factr = 1.0)

    # Recover primal solution
    w_hat = np.sum(optAlpha * Z * D_hat, axis = 1)

    # Compute extended data matrix for test set
    T_hat = np.concatenate((DTE, K * np.array(np.ones([1, DTE.shape[1]]))))
    
    # Compute scores, predictions and accuracy
    S = np.dot(w_hat.T, T_hat)
    PredictedLabels = np.zeros(DTE.shape[1])
    t = 0.0
    PredictedLabels[S >= t] = 1
    acc = sum(PredictedLabels == LTE) / LTE.shape[0]
    
    # Compute duality gap
    obj = 1 - Z * np.dot(w_hat.T, D_hat)
    obj[obj < 0.0] = 0.0
    primal_obj = 1/2 * (w_hat*w_hat).sum() + C * sum(obj)
    gap = primal_obj + dual_obj 

    return primal_obj, abs(dual_obj), abs(gap), 1 - acc


def kernel_SVM(DTR, LTR, DTE, LTE, C, type, d = 0, c = 0, gamma = 0, csi = 0):
        
    # Compute H_hat
    Z = np.ones(LTR.shape)
    Z[LTR == 0] = -1
    ZiZj = np.dot(Z.reshape([Z.shape[0], 1]), Z.reshape([Z.shape[0], 1]).T)
    H_hat = ZiZj * kernel(DTR, DTR, type, d, c, gamma, csi)

    # Optimize the objective function
    obj_function_gradient = obj_function_gradient_wrapper(H_hat)
    B = np.zeros([DTR.shape[1], 2])
    B[:, 1] = C
    optAlpha, dual_obj, _ = fmin_l_bfgs_b(obj_function_gradient, np.zeros(DTR.shape[1]),
                                        approx_grad = False, bounds = B, factr = 1.0)
    
    # Compute scores, predictions and accuracy
    S = np.sum((optAlpha * Z).reshape([DTR.shape[1], 1]) * kernel(DTR, DTE, type, d, c, gamma, csi), axis = 0)
    PredictedLabels = np.zeros(DTE.shape[1])
    t = 0.0
    PredictedLabels[S >= t] = 1
    acc = sum(PredictedLabels == LTE) / LTE.shape[0]

    return abs(dual_obj), 1 - acc


if __name__ == "__main__":

    D, L = load("../Data/Train.txt")  
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)


    """
    Linear SVM applied to a binary task
    """

    Kval = [1, 10]
    Cval = [0.1, 1.0, 10.0]

    print("\n K        C          Primal Loss        Dual Loss        Duality Gap      Error rate  \n")

    for K in Kval:
        for C in Cval:
            primal_obj, dual_obj, gap, err = linear_SVM(DTR, LTR, DTE, LTE, C, K)
            primalJformat = "{:6e}".format(primal_obj)
            dualJformat = "{:6e}".format(dual_obj)
            gapformat = "{:6e}".format(gap)

            print("%2d       %4.1f        %s      %s      %s      %.4f" % (K, C, primalJformat, dualJformat, gapformat, err))

    print("\n")


    """
    Kernel SVM applied to binary task
    """

    csival = [0.0, 1.0]
    C = 1.0
    kernel_type_val = ["poly", "RBF"]
    d = 2
    cval = [0, 1]
    gamma_val = [1.0, 10.0]

    print("\ncsi        C             Kernel                 Dual Loss      Error rate  \n")

    for kernel_type in kernel_type_val:
        if (kernel_type == "poly"):
            for c in cval:
                for csi in csival:
                    dual_obj, err = kernel_SVM(DTR, LTR, DTE, LTE, C, kernel_type, d = d, c = c, csi = csi)
                    dualJformat = "{:6e}".format(dual_obj)
                    kernel_type_string = kernel_type + " (d = " + str(d) + ", c = " + str(c) + ")"
                    print("%.1f       %.1f        %s      %s      %.4f" % (csi, C, "{:<20}".format(kernel_type_string), dualJformat, err))

        else:
            for csi in csival:
                for gamma in gamma_val:
                    dual_obj, err = kernel_SVM(DTR, LTR, DTE, LTE, C, kernel_type, gamma = gamma, csi = csi)
                    dualJformat = "{:6e}".format(dual_obj)
                    kernel_type_string = kernel_type + " (gamma = " + str(gamma) + ")  "
                    print("%.1f       %.1f        %s      %s      %.4f" % (csi, C, "{:<20}".format(kernel_type_string), dualJformat, err))

    print("\n")




