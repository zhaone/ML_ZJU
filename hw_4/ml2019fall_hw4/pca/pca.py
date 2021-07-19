import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    cov = np.cov(data.T)
    eigvalue, eigvector = np.linalg.eig(cov)
    idx = eigvalue.argsort()[::-1]

    return eigvector[:, idx], eigvalue[idx]
    # end answer
