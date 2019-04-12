# -*- coding: utf-8 -*-
"""Created on Sat Mar 25 21:18:59 2017 @author: andres.

This module contains activation functions that are used during propagation
and training of the network. Each one has its normal operating mode and a
derivative mode to be used during training.
"""
__all__ = ['sigmoid','tanh']
#import numpy as np
from netbuilder import np

def sigmoid(x, derivative = False):
    """Implements the sigmoid function, applying it element wise on an array x.

    Parameters
    ----------
    x : numpy array
        This array contains arguments for the sigmoid function.
    derivative : bool
        Indicates whether to use the sigmoid function or its derivative.

    Returns
    -------
    numpy array
        An array of equal shape to `x`.

    """

    if derivative:
        sgm = sigmoid(x)               #Computes the output of the sigmoid function because it is used in its own derivative
        return sgm*(1-sgm)
    else:
        return 1/(1+np.exp(-x))

def tanh(x, derivative=False):
    """Implements the hyperbolic tangent function element wise over an array x.

    Parameters
    ----------
    x : numpy array
        This array contains arguments for the hyperbolic tangent function.
    derivative : bool
        Indicates whether to use the hyperbolic tangent function or its derivative.

    Returns
    -------
    numpy array
        An array of equal shape to `x`.

    """

    if derivative:
        tanh_not_derivative = tanh(x)
        return 1.0 - tanh_not_derivative**2
        #return 1.0 - x**2
    else:
        return np.tanh(x)
