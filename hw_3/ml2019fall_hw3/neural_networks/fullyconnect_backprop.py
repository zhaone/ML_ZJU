import numpy as np

def fullyconnect_backprop(in_sensitivity, in_, weight):
    '''
    The backpropagation process of fullyconnect
      input parameter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: 
                          : [number of images, number of inputs in feedforward]
          weight          : the weight matrix of this layer, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]

      output parameter:
          weight_grad     : the gradient of the weights, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]
          bias_grad       : the gradient of the bias, shape: 
                          : [number of outputs in feedforward, 1]
          out_sensitivity : the sensitivity to the lower layer, shape: 
                          : [number of images, number of inputs in feedforward]

    Note : remember to divide by number of images in the calculation of gradients.
    '''

    # TODO
    # in_sensitivity = (in_ * weight + b)
    N, _ = in_.shape
      
    # begin answer
    weight_grad = np.dot(in_.T, in_sensitivity)/N
    
    bias_grad = np.mean(in_sensitivity, axis=0).reshape(-1, 1)
    
    out_sensitivity = np.dot(in_sensitivity, weight.T)
    # end answer
    
    # print(weight_grad.shape, bias_grad.shape, out_sensitivity.shape)

    return weight_grad, bias_grad, out_sensitivity

