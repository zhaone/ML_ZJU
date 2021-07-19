import numpy as np
import random

def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    n, p = x.shape
    min_, max_ = np.min(x), np.max(x)
    # ctrs = np.random.random((k, p)) * (max_ - min_) + min_
    ctrs = x[random.sample(list(range(n)), k), :]
    iter_ctrs = []
    while True:
        last_ctrs = np.copy(ctrs)
        iter_ctrs.append(last_ctrs)
        idx = _upd_idx(x, ctrs, k)
        ctrs = _upd_ctrs(x, idx, k, last_ctrs)
        if (ctrs == last_ctrs).all():
            break
    iter_ctrs = np.array(iter_ctrs)
    # end answer
    
    return idx, ctrs, iter_ctrs

def _upd_idx(x, ctrs, k):
    '''
    x [n, p]
    ctrs [k, p]
    '''
    n, p = x.shape
    x_ext = np.repeat(x, k, axis=0) # x [kn, p]
    ctrs_ext = np.tile(ctrs, (n, 1))
    sub = x_ext - ctrs_ext
    dist = np.sum(sub**2, axis=1).reshape(n, k)
    idx = np.argmin(dist, axis=1)
    return idx

def _upd_ctrs(x, idx, K, last_ctrs):
    ctrs = []
    for k in range(K):
        mask = idx==k
        if (False == mask).all():
            ctrs.append(last_ctrs[k])
            continue
        x_k = x[mask, :]
        ctrs.append(np.mean(x_k, axis=0))
    return np.vstack(ctrs)