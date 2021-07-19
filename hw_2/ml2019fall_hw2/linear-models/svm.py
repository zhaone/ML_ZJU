import numpy as np
from scipy.optimize import minimize, LinearConstraint

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    X = np.vstack((np.ones((1, N)), X))
    # begin answer
    cons = _get_cons(X, y)
    res = minimize(_objective, np.squeeze(w), method='SLSQP', constraints=cons)
    if not res.success:
        print('failed to give the res, for:', res.message)
        w, num = None, None
    else:
        eps = 10e-5
        w = res.x.reshape((-1, 1))
        dists = np.multiply(y.T, np.dot(X.T, w))
        sc_mask = np.abs(dists-1) < eps
        num = np.sum(sc_mask)
    # end answer
    return w, num

def _objective(w):
    w.reshape((-1, 1))
    return 0.5*np.dot(w.T, w)

def _get_cons(X, Y):
    # X is (p+1,N)
    X = X*np.squeeze(Y)
    lb = np.ones(Y.shape[1])
    ub = np.ones(Y.shape[1])*np.inf
    cons = LinearConstraint(X.T, lb=lb, ub = ub)
#     cons = []
#     for x, y in zip(X.T, np.squeeze(Y)):
#         cons.append({'type': 'ineq', 'fun': lambda w:  y*np.dot(x, w)-1})
    return cons
        