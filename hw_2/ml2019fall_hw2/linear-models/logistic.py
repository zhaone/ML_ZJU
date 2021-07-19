import numpy as np
from copy import copy

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # config
    lrs = [10e-3, 10e-2, 10e-1]
    bz = 5
    epoch = 1000
    stop_th = 0.01
    # begin answer
    X = np.vstack((np.ones((1, N)), X))
    last_err = 1
    for e in range(epoch):
        for i in range(N//bz):
            w = _update_w(X[:, bz*i:bz*(i+1)], y[:, bz*i:bz*(i+1)], w)
            err = get_err_log(X, y, w)
            if abs(last_err-err) < stop_th:
                print('finish')
                break
        else:
            continue
        break
    # end answer
    return w

def _calc_h(X, w):
    '''
    brif: clac h_theta(X), w is P-by-1 mat
    '''
    # clac h_theta(X)
    exp = np.dot(w.T, X)
    return 1/(1+np.exp(-exp)).reshape((1, -1))

def _update_w(X, y, w, lr=10e-3):
    tmp_y = copy(y)
    tmp_y[tmp_y==-1] = 0
#     print("X", X)
#     print("tmp_y", tmp_y)
#     print("old w", w)
    h_t = _calc_h(X, w)
    #print("h_t", h_t)
    grad = np.mean(X*np.squeeze(h_t-tmp_y), axis=1)
    #print("grad", grad)
    w -= lr*grad.reshape((-1, 1))
    #print("new w", w)
    return w

def get_err_log(X, y, w, th=0.5):
    h_t = _calc_h(X, w)
    pred = np.sign(h_t-0.5)
    pred[pred==0] = -1
    #print("pred", pred)
    #print("label", y)
    mask = pred == y
    right_num = np.sum(mask)
    acc = right_num/X.shape[1]
    return 1-acc
    