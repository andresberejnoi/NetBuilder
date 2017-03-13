# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 -0.52:-0.52:50.5 2-0.50.56

@author: andresberejnoi
"""

from ..NeuralNet import Network, save_outputs
#from tools import save_outputs
import numpy as np

'''
Because of the tanh functions used by default for the activations of the network,
better results are obtained by using -0.5 instead of 0 and 0.5 instead of 1. So, for example,
the truth table for the AND gate will be as:
    -0.5 -0.5 | -0.5
    -0.5  0.5 | -0.5
     0.5 -0.5 | -0.5
     0.5  0.5 |  0.5
    
Therefore it is important to remember that when interpreting the result of the network
'''


# Teach network XOR function
#training set for XOR
xor = [
        [np.array([-0.5,-0.5]), np.array([-0.5])],
        [np.array([-0.5,0.5]), np.array([0.5])],
        [np.array([0.5,-0.5]), np.array([0.5])],
        [np.array([0.5,0.5]), np.array([-0.5])]
      ]

#training set for logic AND 
aand = [
         [np.array([-0.5,-0.5]), np.array([-0.5])],
         [np.array([-0.5,0.5]), np.array([-0.5])],
         [np.array([0.5,-0.5]), np.array([-0.5])],
         [np.array([0.5,0.5]), np.array([0.5])]
       ]
    
#training set for logic OR 
oor = [
        [np.array([-0.5,-0.5]), np.array([-0.5])],
        [np.array([-0.5,0.5]), np.array([0.5])],
        [np.array([0.5,-0.5]), np.array([0.5])],
        [np.array([0.5,0.5]), np.array([0.5])]
      ]

# The above training sets are not exactly in the correct format that the network expects, but I do not want to rewrite them again,
# so below they will be "fixed"
xor = [(array[0],array[1]) for array in xor]
aan = [(array[0],array[1]) for array in aand]
oor = [(array[0],array[1]) for array in oor]

#Setting up the network
topology = [2,10,10,1]
epochs = 1000
tolerance = 1E-10
trainingSet = xor                           #change this to any of the training sets above: trainingSet = aand, etc

print("="*80)
print("Training...\n")
net = Network(topology, learningRate=0.1, momentum=0.1)
net.train(trainingSet,epochs,tolerance, batch=False)         #training begins

#Now, show the results of training
#It would be better to create a function to display this information in a better way
#print("="*80)      #will 80 '=' signs to separate the line
print()
print("Testing network:")
print("INPUTS    |\tPREDICTION\t   | EXPECTED")
for inputs,target in trainingSet:
    out = net.feedforward(inputs)

    print("{0} {1} \t {2} \t\t\t {3}   ".format(inputs[0],inputs[1],out[0],target[0]))              #for some reason, the last line is not tabbed in
    
print("="*80)


#extracting the value of the training pattern:
xor_patterns = [pat[0] for pat in xor]
save_outputs("xor_outs.csv", xor_patterns, net)
