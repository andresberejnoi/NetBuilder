# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:24:00 2017

@author: andres
"""
import numpy as np
def mean_squared_error(target,actual,derivative=False):
    """A simple loss function. It computes the difference between target and actual and raises the value to the power of 2, and everything is divided by 2. The computed value is the error.
 
    target: numpy array with values we want the network to approximate
    actual: numpy array (same shape as target); the output of the network after feedforward
    
    return: error
    """
    try:
        assert(target.shape==actual.shape)
    except AssertionError:
        print("""Shape of target array '{0}' does not match shape of actual '{1}'""".format(target.shape,actual.shape))
        raise
    if not derivative:
        #compute the error and return it    
        #print('='*80)
        #print('Error Function: MSE\nTarget:\tActual:')
        #for i in range(len(target)):
        #    print(target[i],actual[i])
        
        #print()
        #print('Summing over rows and squaring:')
        #for i in range(len(target)):
        #    print(np.sum((target[i]-actual[i])**2))
        error = np.sum(0.5 * np.sum((target-actual)**2, axis=1, keepdims=True))
        return error
    else:
        return (actual - target)
