#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Carlos Andres Berejnoi Bejarano
#
# Created:     30/12/2015
# Copyright:   (c) PcNueva CoreI7 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np

class trainer(object):
    """
    A collection of methods that help in training a neural network. Currently, it
    will be desgined to train a feedforward neural network using backpropagation.

    If there is time, online and batch training will be both implemented.
    """

    def __init__(self):
        """"""
        self.trainingSet_inputs = []                            # it will contain the training inputs to be fed into the network
        self.trainingSet_targets = []                           # it will contain the training outputs to be used for the backpropagation step
        self.threshold = 1E-5                                      # the error threshold to tell the training when the fitness of the network is good enough
        self.trainingMethod = "online"
        #self.net = None                                         # will contain the neural network to be trained


    def _outputError(self,target,actual):
        """
        calculates the error
        Error = E = 1/2 * (target - output)^2
        """

        difference = target - actual               # the difference between expected output and actual output
        squared = np.power(difference,2) / 2

        return squared

    def _error_total(self, errorVector):
        """
        Error total is the sum of the errors in each output neuron.
        E_total = E1 + E2 + E3 + ... + En"""
        E_total = np.sum(errorVector)                           # The error of all the output neurons are combined

        return E_total

    def computerError(self,targets,actuals):
        """
        Computes the error in the output of the network.
        """
        vectorError = self._outputError(targets,actuals)
        E_total = self._error_total(vectorError)
        return E_total


    def train(self, net, epochs = 1000):
        """
        Trains the network using backpropagation.
        net: a Neural Network object. It is the one that needs to be trained.
        epochs: an int. The number of times backpropagation will be carried out if a good enough network is not found before
        """
        trainingSet = zip(self.trainingSet_inputs,self.trainingSet_targets)

        print("Training Set: ",trainingSet)

        # The following loop implements the online training method for teaching the network
        for i in range(epochs):
            for I,T in trainingSet:                         # I is a numpy array corresponding to one training example input vector to the network
                                                            # T is a numpy array corresponding to the target output vector for training example I
                out = net.feedforward(I)
                Error_total = self.computerError(T,out)                          # gets a numerical value of the error in the network


                #print("Network Output: ", out)
                #print("Error: ",Error_total)

                if Error_total < self.threshold:                                    # compares to the error threshold. THIS NEEDS IMPROVEMENT/CLEARER CODE
                    break
                else:
                    net.backprop_EXPERIMENTAL(T,out)                                             # if the error is still too high, backpropagate it to reduce it


        return net                 # returns the improved network

    def batch_training(self,net,epochs):
        """Implements batch training
        NOT WORKING YET"""
        trainingSet = zip(self.trainingSet_inputs,self.trainingSet_targets)

        print("Training Set: ",trainingSet)
        errors = []                             # this will contain the errors of the training set


        i = 0
        while i < epochs:
            for I,T in trainingSet:
                out = net.feedforward(I)
                errors.append(self.computerError(T,out))


    def train_EXP(self,net,epochs=100000):
        """
        Tries to fix the problem with the official trainer
        """
        trainingSet = list(zip(self.trainingSet_inputs,self.trainingSet_targets))

        #print("Training Set: ",trainingSet[0],trainingSet[1])
        #print()
        
        for i in range(epochs):
            error = net.trainEpoch(trainingSet)
         #   print()
         #   print("FIRST ERROR:",error)
         #   print()
            
            if i % 10000 == 0:
                self.print_stateOfTraining(i,error,self.threshold)
                
            if error <= self.threshold:
                print("Final epoch:",i)
          #      print("Is error low?: ",error<=self.threshold)
                print("Final Error: ",error)
                self.print_trainingComplete()
                #return net
                break
            
                
        return net
                
                
                



    #printers 
    def print_trainingComplete(self):
        """Prints a message indicating that the training has been completed"""
        print("Network has reached a state of minimum error.")

    def print_stateOfTraining(self,iterCount,error,threshold):
        """Prints the current state of the training process, such as the epoch, current error"""
        print("Epoch:",iterCount)
        print("Error:",error,"Threshold:",threshold)
        
    # Setters:
    def set_trainingInputs(self,inputs):
        """
        Sets the variable self.trainingSet_inputs to a Python list passed as argument.
        inputs: a Python list containg all the inputs of the training set.

        returns: None
        """
        self.trainingSet_inputs = inputs

    def set_trainingTargets(self,targets):
        """
        Sets the variable self.trainingSet_targets to a Python list passed as argument.
        targets: a Python list containg all the target outputs of the training set.

        returns: None
        """
        self.trainingSet_targets = targets

    def set_threshold(self,threshold):
        """
        Sets the threshold for the error function in the training process.
        threshold: a numerical value for the threshold.

        returns: None
        """
        self.threshold = threshold
