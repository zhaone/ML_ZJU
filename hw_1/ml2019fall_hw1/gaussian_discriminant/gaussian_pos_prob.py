import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    M = X.shape[0]
    for k in range(K):
        mu = Mu[:, k]
        sigma = Sigma[:, :, k]
        phi = Phi[k]
        # transpose of X - mu
        XT = X.T- mu
        # inverse mat of sigma
        si = np.linalg.inv(sigma)
        # dominat of sigma
        sd = np.linalg.det(sigma)
        
        pol = np.multiply(np.dot(XT,si).flatten('C') , XT.flatten('C'))
        pol = np.sum(pol.reshape(-1, M), axis=1)
        
        posterior_k = np.exp(-0.5 * pol)/ (2*np.pi*np.sqrt(sd))
        
        p[:, k] = posterior_k
    p = (p*Phi)
    p_x = np.sum(p, axis=1)
    p = (p.T / p_x).T
    
    # end answer
    
    return p
    