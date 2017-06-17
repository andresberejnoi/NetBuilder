#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:25:09 2017

@author: andresberejnoi
"""

from . import *    #this will import everything necessary
#from . import Network, mean_squared_error
#from . import tanh,sigmoid
#from . import save_model,load_model

import numpy as np

#Some tests
def random_training_set():
    
    #Define input and output layer neurans
    numIn = 2
    numOut = 3
    #create random inputs and outputs
    np.random.seed(50)
    input_set = np.random.rand(1000,numIn)   #1000 samples where each sample has numIn features
    target_set = np.random.rand(1000,numOut)   
    
    net = Network(topology=[numIn,3,numOut])
    net.train(input_set=input_set,
              target_set=target_set,
              batch_size=0,
              epochs=1000)
    
def test_AND():
    
    print('='*80)
    print("TEST AND")
    #Define input and output layer neurans
    numIn = 2
    numOut = 1
    
    #num_samples = 4
    
    #Create training sets
    T,F = 1.,-1.
    input_set = np.array([[F,F],
                          [F,T],
                          [T,F],
                          [T,T]])
    
    target_set = np.array([[F],
                           [F],
                           [F],
                           [T]])

    
    net_name = 'AND'
    net = Network()
    net.init(topology=[numIn,numOut],name=net_name)
    #net.set_outActivation_fun(func='sigmoid')
    net.train(input_set=input_set,
              target_set=target_set,
              batch_size=0,
              epochs=1000,
              print_rate=100)
    
    x = input_set[0:1]
    y = target_set[0:1]
    print(x.shape)
    test_out = net.feedforward(x)
    print('TEST OUTPUT:')
    print(test_out)
    
    error = mean_squared_error(target=y,actual=test_out)
    print('ERROR:',error)
    print('='*80)
    
    return net

def test_XOR():
    
    print('='*80)
    print("TEST XOR")
    #Define input and output layer neurans
    numIn = 2
    numOut = 1
    
    #num_samples = 4
    
    #Create training sets
    T,F = 1.,-1.
    input_set = np.array([[F,F],
                          [F,T],
                          [T,F],
                          [T,T]])
    
    target_set = np.array([[F],
                           [T],
                           [T],
                           [F]])

    
    net_name = 'XOR'
    net = Network()
    net.init(topology=[numIn,5,numOut],name=net_name)
    net.train(input_set=input_set,
              target_set=target_set,
              batch_size=4,
              epochs=1000,
              print_rate=100)
    
    
    test_out = net.feedforward(input_set)
    print('\nTEST OUTPUT:')
    print(test_out)
    
    error = mean_squared_error(target=target_set,actual=test_out)
    print('ERROR:',error)
    print('='*80)
    
    return net
    
if __name__=='__main__':
    #random_training_set()
    net = test_AND()
    #net = test_XOR()
    
    #Test saving method
    output_path = save_model(net=net)
    net_loaded = load_model(output_path)
    
