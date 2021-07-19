import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer
    
    # add bias
    X = np.vstack((np.ones((1, N)), X))
    y = np.squeeze(y)
    # begin iter
    last_iter = iters
    while True:
        last_iter = iters
        for n in range(N):
            out = np.dot(X[:, n], w)
            if y[n]*out[0] <= 0:
                w = (w.T + y[n]*X[:, n]).T
                iters += 1
        if last_iter == iters: # no update anymore
            break
    # end answer
    
    return w, iters