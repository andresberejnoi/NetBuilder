# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:18:59 2017

@author: andres
"""
import numpy as np

def sigmoid(x, derivative = False):
    """Implements the sigmoid function, applying it element wise on an array x.

    x: numpy array with arguments for the sigmoid function.
    derivative: a boolean indicating whether to use the sigmoid function or its derivative.
    """

    if derivative:
        sgm = sigmoid(x)               #Computes the output of the sigmoid function because it is used in its own derivative
        return sgm*(1-sgm)
    else:
        return 1/(1+np.exp(-x))
        
def tanh(x, derivative=False):
    """Implements the hyperbolic tangent function element wise over an array x.

    x: numpy array with arguments for the hyperbolic tangent function.
    derivative: a boolean value indicating whether to use the tanh function or its derivative.
    """

    if derivative:
        tanh_not_derivative = tanh(x)
        return 1.0 - tanh_not_derivative**2
        #return 1.0 - x**2
    else:
        return np.tanh(x)
