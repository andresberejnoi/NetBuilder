# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 06:22:09 2016

@author: andresberejnoi
"""
import numpy as np
from NeuralNet import *

x1 = np.array([0,0])
x2 = np.array([0,1])
x3 = np.array([1,0])
x4 = np.array([1,1])


trainingInputs = [x1,x2,x3,x4]
trainingTargets = [np.array([0]),
                    np.array([1]),
                    np.array([1]),
                    np.array([0])]
                    

trainingSet = list(zip(trainingInputs,trainingTargets))
epochs = 10000
error = 0
tolerance = 1E-5

net = network([2,4,1])
net.setup()
net.set_learningRate(0.1)

for i in range(epochs+1):
    error = net.trainEpoch(trainingSet)
    
    if i %(epochs/10) == 0:
        print("Iteration {0}\nError: {1}\tThreshold: {2}".format(i,error,tolerance))
    
    if error <= tolerance:
        print("Minimum error reached.")
        print("Iteration: {0}\nFinal Error: {1}")
        break
    
print("All {0} iterations completed".format(epochs))

print(net.feedforward(x2))

    