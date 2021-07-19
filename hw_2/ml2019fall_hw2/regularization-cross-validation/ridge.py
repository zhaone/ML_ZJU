import numpy as np

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    X = np.vstack((np.ones((1, N)), X))
    y = y.reshape((-1,1))
    
    unit = np.eye(P+1)
    if lmbda == 0:
        inv = np.linalg.pinv(np.dot(X, X.T) + lmbda*unit)
    else:
        inv = np.linalg.inv(np.dot(X, X.T) + lmbda*unit)
    w = np.dot(np.dot(inv, X), y)
    # end answer
    return w
