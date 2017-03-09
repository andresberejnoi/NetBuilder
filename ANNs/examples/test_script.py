# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:25:57 2016

@author: andresberejnoi
"""
import numpy as np
from NeuralNet import Network, sigmoid


        
#Testing the network
#np.random.random()
#np.random.normal(scale=0.1,size=(4,4))                                          #attempt to change the random numbers, or something
        
# Training exmaples for XOR gate
'''
x1 = np.array([0,0])
x2 = np.array([0,1])
x3 = np.array([1,0])
x4 = np.array([1,1])

trainingTargets = [np.array([0]),
                    np.array([1]),
                    np.array([1]),
                    np.array([0])]
'''

x1 = np.array([-0.5,-0.5])
x2 = np.array([-0.5,0.5])
x3 = np.array([0.5,-0.5])
x4 = np.array([0.5,0.5])

trainingInputs = [x1,x2,x3,x4]     #                                            # an array of the inputs                             
# Targets for training the network forn XOR gate

       
trainingTargets = [np.array([-0.5]),
                    np.array([0.5]),
                    np.array([0.5]),
                    np.array([-0.5])]

'''               
trainingTargets = [np.array([0]),
                    np.array([1]),
                    np.array([1]),
                    np.array([0])]
'''
                    
        
        
topology = [2,10,10,1]
net = Network(topology,0.1,0.1)
net.save("recog_number.csv", transpose=True, keep_bias = False)                 # saving a file with the iniitial weights
#net.outActiv_fun = sigmoid


trainingSet = list(zip(trainingInputs,trainingTargets))
epochs = 10000
tolerance = 1E-10

print("Initial Weights:")
for W in net.weights:
    print(W)
    print()

net.train(trainingSet,epochs,tolerance)


def test(rep=10):
    '''A small test function'''
    global trainingSet
    topology = [2,5,5,1]
    net = Network(topology)
    net.Gradients = [None,None]
    for i in range(rep+1):
        error=0.0
        for I,P in trainingSet:
            print("Weights:")
            print(net.weights[0])
            print("Gradients:")
            print(net.Gradients[0])
            #print("Previous change:")
            #print(net.last_change[0])
            print()
            print(net.weights[1])
            print("Gradients:")
            print(net.Gradients[1])
            #print("Previous Change:")
            #print(net.last_change[1])
            print()
    
            error += net.backprop(I,P)
            
            print("Activations:")
            print("L=0 : ",net.netOuts[0])
            print("L=1 : ",net.netOuts[1])
            print("L=2 : ",net.out)
            print()
        
        print("-----------------------------------")
        print("ERROR: ",error, "EPOCH: ",i)
        print("-----------------------------------")
        print()
