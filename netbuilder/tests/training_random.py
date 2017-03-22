#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 02:12:46 2017

@author: andresberejnoi
"""

import numpy as np
from .. import Network

def random_training_set():
    
    #Define input and output layer neurans
    numIn = 5
    numOut = 3
    #create random inputs and outputs
    np.random.seed(50)
    input_set = np.random.rand(1000,numIn)   #1000 samples where each sample has numIn features
    target_set = np.random.rand(1000,numOut)   
    
    net = Network(topology=[numIn,15,15,numOut])
    net.train(input_set=input_set,
              target_set=target_set,
              batch_size=100)
    