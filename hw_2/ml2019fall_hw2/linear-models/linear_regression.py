import numpy as np

def _one_hot(y, num_class):
    y[y==-1] = 0
    y = np.squeeze(y).astype(int)
    return np.eye(num_class)[y]

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    one_hot = _one_hot(y, 2) # fix the class num 2
    X = np.vstack((np.ones((1, N)), X))
    inv = np.linalg.inv(X.dot(X.T))
    w = np.dot(np.dot(inv, X), one_hot)
    # end answer
    return w
