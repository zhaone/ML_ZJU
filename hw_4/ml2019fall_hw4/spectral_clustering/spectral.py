import numpy as np
from kmeans import kmeans


def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    D = np.sum(W, axis=1)
    L = np.diag(D) - W
    D_sqrt = np.diag(D**(-0.5))
    Lap_mat = D_sqrt@L@D_sqrt
    e, v = np.linalg.eig(Lap_mat)
    idx = e.argsort()
    v = v[:, idx].real
    # print(type(L), type(D_sqrt), type(Lap_mat), type(v))
    fea = v[:, :k]
    idx = kmeans(fea, k)
    return idx.reshape(-1), v
    # end answer
