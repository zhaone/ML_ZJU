import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from pca import PCA

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer
    # get main dir
    hs, ws = np.nonzero(img_r[:, :, -1])
    pixs = np.vstack((hs, ws)).T
    v, e = PCA(pixs)
    main_dir = v[:, 0]
    # rotate
    theta = -np.arctan2(main_dir[1], main_dir[0]) * 180 / np.pi - 90
    print('main_dir:', main_dir, 'theta:', theta)
    img = ndimage.rotate(img_r, theta)

    return img.astype(np.int)
    # end answer
