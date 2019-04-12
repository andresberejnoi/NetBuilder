# -*- coding: utf-8 -*-
"""Created on Sat Mar 25 21:24:00 2017 @author: andres

This module contains error/loss functions that can be used to train the network

"""
__all__ = ['mean_squared_error']
#import numpy as np
from netbuilder import np

def mean_squared_error(target,actual,derivative=False):
    """A simple loss function. It computes the difference between target and actual and
    raises the value to the power of 2, and everything is divided by 2. The computed value is the error.

    Parameters
    ----------
    target : numpy array
        Contains the values we want the network to approximate.
    actual : numpy array
        Same shape as target. It is the output of the network after feedforward propagation.

    Returns
    -------
    float/numpy array
        If function is called in normal mode, then a float (the error) is returned. When derivative mode
        is used, a numpy array is returned (same shape as input arrays).
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
