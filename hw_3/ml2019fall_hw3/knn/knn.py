import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    print(x.shape)
    N_test, _ = x.shape
    N_train, _ = x_train.shape
    
    y = np.zeros(N_test)
    # compute similarity
    test = np.repeat(x, N_train, axis=0) # N_train * N_test, P
    train = np.tile(x_train, (N_test, 1))
    sub = test-train
    sim = np.sum(sub**2, axis=1)
    sim = sim.reshape(N_test, N_train)

    idx = sim.argsort(axis=1)[:, :k].reshape(-1)# N_test * k
    neis = y_train[idx].reshape(-1, k)
    for i, nei in enumerate(neis):
        y[i] = np.argmax(np.bincount(nei))
    # end answer

    return y
