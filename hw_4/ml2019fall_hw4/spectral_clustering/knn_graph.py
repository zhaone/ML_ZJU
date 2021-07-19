import numpy as np


def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    n, p = X.shape
    # get all knn index
    left = np.repeat(X, n, axis=0)
    right = np.tile(X, (n, 1))
    sub = left-right
    dis = np.sum(sub**2, axis=1)
    dis = dis.reshape(n, n)
    col_idx = dis.argsort(axis=1)[:, :k]
    row_idx = np.repeat(np.arange(n).reshape((n, 1)), k, axis=1)

    # compute wij
    W = np.zeros((n, n))
    var = np.sum(np.var(X, axis=0))
    dis_hk = np.exp(-dis[row_idx, col_idx]/(2*var))
    dis_hk[dis_hk < threshold] = 0  # set wij < threshold = 0
    W[row_idx, col_idx] = dis_hk

    return W
    # end answer
