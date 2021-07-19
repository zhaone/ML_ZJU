import numpy as np

def relu_backprop(in_sensitivity, in_):
    '''
    The backpropagation process of relu
      input paramter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: same as in_sensitivity
      
      output paramter:
          out_sensitivity : the sensitivity to the lower layer, shape: same as in_sensitivity
    '''
    # TODO

    # begin answer
    in_ [in_ > 0] = 1 #
    in_ [in_ < 0] = 0 #
    out_sensitivity = np.multiply(in_sensitivity, in_)
    
    # end answer
    return out_sensitivity

