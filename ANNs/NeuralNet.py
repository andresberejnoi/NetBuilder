# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:37:08 2016

@author: andresberejnoi
"""
import numpy as np

def sigmoid(x, derivative = False):
    """
    Implements the sigmoid function, applying it element wise on an array x.
    derivative: a boolean indicating whether to use the sigmoid function or its derivative
    """
    if derivative:
        sgm = sigmoid(x)               #Computes the output of the sigmoid function because it is used in its own derivative
        return sgm*(1-sgm)
    else:
        return 1/(1+np.exp(-x))
        
def tanh(x, derivative=False):
    """
    Implements the hyperbolic tangent function element wise over an array x.
    derivative: a boolean value indicating whether to use the tanh function or its derivative."""
    if derivative:
        tanh_not_derivative = tanh(x)
        return 1.0 - tanh_not_derivative**2
        #return 1.0 - x**2
    else:
        return np.tanh(x)
        

class network(object):
    """
    Implements the methods and attributes of an Artificial Neural Network (Multilayer)
    """
    #np.random.seed()                       # start the random seed before using it
    np.random.random()
    def __init__(self,topology,learningRate=0.1, momentum=0.1):
        self.topology = topology
        self.size = len(topology)-1
        self.learningRate = learningRate
        self.momentum = momentum
        
        self.functions = {'tanh':tanh,'sigmoid':sigmoid}
        
        # Initialize random weights, and create empty matrices to store the previous changes in weight (for momentum):
        self.weights = []
        self.last_change = []
        for i in range(len(topology)-1):
            self.weights.append(np.random.normal(loc=0,scale=0.1,size=(topology[i+1],topology[i]+1)))
            self.last_change.append(np.zeros( (topology[i+1],topology[i]+1) ))
                    
        # Initialize the activation functions
        self.outActiv_fun = tanh
        self.hiddenActiv_fun = tanh
         
            
        
    # Overloading Operators:
    def __str__(self):
        return "Network: {0}".format(self.topology)
    
    __repr__ = __str__
        
        
    #
    # Section below is for setters
    #
    def set_hiddenactivation_fun(self,func='tanh'):
        try:
            self.hiddenActiv_fun = self.functions[func]
        except:
            message = """The function '{0}' is not available.\nPlease select one of the following functions:\n{1}""".format(func, ''.join(['-> '+fun+'\n' for fun in list(self.functions)]) )
            print(message)
            #raise KeyError
        

    #
    # Functionality of the network
    #
    def feedforward(self,inputs, batch=False):
        """
        Performs the feedforward propagation of the inputs through the layers.
        inputs: a vector of input for the layer
        batch: a boolean value to decide whether to treat the input as batch or online training. Batch not implemented yet
        """
        self.netIns = []
        self.netOuts = []
        
        if not batch:
            I = np.append(inputs,[1])                                           # adds the bias input of 1
            self.netOuts.append(I)
            
            for idx in range(self.size):
                W = self.weights[idx]
                
                I = np.dot(W,I)
                self.netIns.append(I)
                
                if idx == self.size -1:
                    I = self.outActiv_fun(I)
                else:
                    I = self.hiddenActiv_fun(I)
                    I = np.append(I,[1])
                    self.netOuts.append(I)
            
            self.out = I
            return self.out

        else:
            """implement batch training"""
            pass
        
    def backprop(self,inputs,target,batch=False):
        """
        Backpropagation (online)
        inputs: a vector of inputs for the neural network. It corresponds to one training example (in online mode)
        target: a vector of expected values correspoding to the inputs vector
        """
        Gradients = [None]*self.size                        # it will have the same size as self.weights
        
        output = self.feedforward(inputs)                                       # performs forward propagation of the inputs 
        
        # Compute the error for the network at this particular example
        error = 0.5 * np.sum((target-output)**2)
        delta = None
        gradients = None 
        
        for i in range(self.size):
            back_index =self.size-1 -i                  # This will be used for the items to be accessed backwards            
            if i==0:
                # First, we calculate the delta for the output layer
                delta = (output-target) * self.outActiv_fun(self.netIns[back_index], derivative=True)
                gradients = np.outer(self.netOuts[back_index], delta).transpose()
                Gradients[back_index] = gradients
            else:
                # The calculation for the hidden deltas is slightly different than the output neurons
                W_with_bias = self.weights[back_index+1]                                  # gets the weight matrix for the layer that was left behind
                W = np.delete(W_with_bias, W_with_bias.shape[1]-1,1)                        # and creates a new matrix without the bias values
                delta = np.dot(delta, W) * self.hiddenActiv_fun(self.netIns[back_index], derivative=True)
                #delta = np.dot(delta, W) * self.hiddenActiv_fun(self.netOuts[back_index], derivative=True)
                gradients = np.outer(self.netOuts[back_index], delta).transpose()
                
                Gradients[back_index] = gradients
            
        self.Gradients = Gradients
        # Update the weights, because this is online training
        for i in range(self.size):
            delta_weight = self.learningRate*Gradients[i]
            self.weights[i] -= delta_weight + self.momentum*self.last_change[i]
            self.last_change[i] = Gradients[i]
            
        return error
    
    
    def trainEpoch(self,trainingSet):
        """
        Presents every training example to the network once, backpropagating the error
        for each one.
        trainingSet: a list of tuples pairing inputs,targets for each training example.
        Returns: cumulative error of the epoch
        """
        epoch_error = 0
        
        for inputs,targets in trainingSet:
            epoch_error += self.backprop(inputs,targets)
        
        return epoch_error
        
    
    def train(self,trainingSet,epochs=10000,threshold_error = 1E-5):
        """
        Trains the network for the specified number of epochs."""
        
        for i in range(epochs+1):
            error = self.trainEpoch(trainingSet)
            
            if i % (epochs/100) == 0:                                            # Every certain number of iterations, information about the network will be printed
                self.print_stateOfTraining(i,error)
            if error <= threshold_error:                                        # when the error of the network is less than the threshold, the traning can stop
                self.print_stateOfTraining(i,error, finished=True)
                break
    
    
    
    # Information printers
    def print_stateOfTraining(self,iterCount,error,finished=False):
        """Prints the current state of the training process, such as the epoch, current error"""
        #print("Epoch:",iterCount)
        if finished:
            print("Network has reached a state of minimum error.")
        print("Error: {0}\tEpoch {1}".format(error,iterCount))
        
    
