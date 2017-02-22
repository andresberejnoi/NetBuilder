# -*- coding: utf-8 -*-
"""
    @author: andresberejnoi
"""
#TODO: During training, if the network gets stuck in a local minima for several epochs,
# then randomly modify certain weights in the matrix. This might allow the network to get out
# of that minima and converge 

import numpy as np
import tools         # this is just a python file where I will put some functions before I decide to include them here directly

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
#---------------------------------------------------------------------------------------------
class NetworkError(Exception):
    '''
    An exception object that can be raised to handle different situations.
    It is currently very simple.    
    '''
    def __init__(self, msg):
        '''
        msg: a string with a message to be displayed when this exception is raised.        
        '''
        self.msg = msg
        
    def __str__(self):
        return self.msg
        
#----------------------------------------------------------------------------------------------

class network(object):
    """
    Implements the methods and attributes of an Artificial Neural Network (Multilayer).
    The network implemented is a feedforward one, with backpropagation as the training algorithm.
    This implementation uses the numpy library, so it needs to be available to be able to run
    """
    #np.random.seed()                       # start the random seed before using it
    #np.random.random()
    #Set up a dictionary of activation functions to access them more easily
    functions = {'tanh':tanh,'sigmoid':sigmoid}

    def __init__(self,topology=[2,5,5,1],learningRate=0.1, momentum=0.1, loadfile=None):
        '''
        topology: A Python list with integers indicating the shape of the network. 
                    i.e: [5,10,1]: this encodes a network of 3 layers (one input, 1 hidden, and 1 output). 
                        The input layer will have 5 neurons, the hidden layer will have 10, and the 
                        output layer will have only one neuron.
        learningRate: a float that helps with the speed and convergence of the network. It is usually small.
                        A very small number will cause the network to converge very slowly. A high rate will make
                        the network oscillate during training and prevent it from "learning" patterns.
        momentum: A float, also used during the training process. It is related to how much the previous changes
                        affect the new ones.
        '''
        
        if loadfile is None:                    # this will be used when the network parameters are provided instead of a file to read from
            self.topology = topology
            self.size = len(topology)-1                                             #The size of the network will be the number of weeight matrices between layers, instead of the number of layers itself
            self.learningRate = learningRate
            self.momentum = momentum
            
                    
            # Initialize random weights, and create empty matrices to store the previous changes in weight (for momentum):
           
            self.weights = [np.random.normal(loc=0,scale=0.6,size=(topology[i]+1, topology[i+1])) for i in range(self.size)]  
            #self.last_change = [np.zeros( (topology[i]+1 , topology[i+1] ) ) for i in range(self.size)]
            #self.weights = [np.random.normal(loc=0,scale=0.1,size=( topology[i+1], topology[i]+1)) for i in range(self.size)]
            #self.last_change = [np.zeros( (topology[i+1],topology[i]+1) ) for i in range(self.size)]
            
            
            '''
            for i in range(len(topology)-1):
                #Every layer has a bias node, so each matrix will have extra weights correspoding to the connections from that bias node
                #The rows of the matrix correspond to neurons in next layer, while columns correspond to nodes in previous layer.
                #i.e. network [5,10,1] will have 2 weight matrices, one from input to hidden, then from hidden to output and 
                # matrix shapes will be: input-to-hidden -> 10x6, hidden-to-out -> 1x11; the +1 on the columns is the result of having a bias node on that layer
                self.weights.append(np.random.normal(loc=0,scale=0.1,size=(topology[i+1],topology[i]+1)))           #weight values are initialized randomly, between -0.1 and 0.1
                self.last_change.append(np.zeros( (topology[i+1],topology[i]+1) ))                                  #creating empty matrices to keep track of previous gradients. this will be used along with the momentum term during backpropagation
                
            '''
        else:                           #when the file is provided:
        
            if '.csv' in loadfile:
                self.topology, self.weights = tools.read_weights(loadfile)
                
            else:
                while True:
                    try:
                        self.weights = np.load(loadfile)
                        break
                        #self.size = len(self.weights)
                    except FileNotFoundError:
                        print("""File '{0}' is was not found.""".format(loadfile))
                        loadfile = input("Please enter an available file (to cancel type 'break'): ")
                        if loadfile.lower() == 'break':
                            raise NetworkError("Could not initialize the network")
                self.topology = [(M.shape[0]-1) for M in self.weights]
                self.topology.append(self.weights[-1].shape[1]) 
            
            # We set up the other varibales
            self.size = len(self.weights)
            self.learningRate = learningRate
            self.momentum = momentum
                        
            
            
        
        # Initialize activation functions.
        self.outActiv_fun = tanh
        self.hiddenActiv_fun = tanh
        self.Gradients = [None]*self.size
         
            
    #  
    # Overloading Operators:
    #
    def __str__(self):
        '''
        For now, the string method simply returns the topology of the network.
        '''
        return "Network: {0}".format(self.topology)
    
    __repr__ = __str__
        
        
    #
    # Section below is for setters
    #
    def set_hiddenactivation_fun(self,func='tanh'):
        '''
        Changes the hidden activation function for a different one, as long as
        the desired function is available in the dictionary self.functions
            
        func: a string with the name of the desired function (it will be the key for the dictionary)
        '''
        try:
            self.hiddenActiv_fun = self.functions[func]
        except:
            message = """The function '{0}' is not available.\nPlease select one of the following functions:\n{1}""".format(func, ''.join(['-> '+fun+'\n' for fun in list(self.functions)]) )
            print(message)
            #raise KeyError
        
    def set_outActivation_fun(self,func='tanh'):
        '''
        Changes the output activation function for a different one, as long as
        the desired function is available in the dictionary self.functions
            
        func: a string with the name of the desired function (it will be the key for the dictionary)
        '''
        try:
            self.outActiv_fun = self.functions[func]
        except:
            message = """The function '{0}' is not available.\nPlease select one of the following functions:\n{1}""".format(func, ''.join(['-> '+fun+'\n' for fun in list(self.functions)]) )
            print(message)
            #raise KeyError

    #
    # Functionality of the network
    #
    
    def save(self, filename, transpose=False, keep_bias = False):
        """
        Saves the weights of the network stored in self.weights using numpy 'save' method.
        
        filename: a string or file object where the information will be saved 
        Transpose: boolean; used when saving to a csv file. It tells whether the matrix should be saved in its
                    current form, or its transpose
        """
        if '.csv' in filename:      # if the file weights are to be saved in csv format:
            handler = open(filename, 'wb')       # opening in byte write mode to match with numpy's opening mode
            np.savetxt(handler,np.array([self.topology]), delimiter=',')        #saves the header for the file         
            
            if transpose:
                for Mat in self.weights:
                    # iterate through every weight matrix and save it to file
                    if keep_bias is False:          # when the user does not want to include the bias vector weight into the weights file
                        Mat = Mat[:-1]              # the last row corresponds to the bias weight vector, so we slice it off
                    np.savetxt(handler, Mat.transpose(), delimiter=',')
            else:
                for Mat in self.weights:
                    if keep_bias is False:
                        Mat = Mat[:-1]
                    np.savetxt(handler, Mat, delimiter=',')
                    
            handler.close()
        else:
            try:
                np.save(filename,self.weights)
                print("Weights were saved successfully")
            except:
                print("There was an error saving the weights")
                
                
    def feedforward(self,inputs, batch=False):
        """
        Performs the feedforward propagation of the inputs through the layers.
        inputs: a vector of inputs for the first layer
        batch: a boolean value to decide whether to treat the input as batch or online training. Batch not implemented yet
        """
        # These two lists will contain the inputs and the outputs for each layer, respectively
        self.netIns = []                                                        
        self.netOuts = []
        
        if not batch:
            I = np.append(inputs,[1])                                           # adds the bias input of 1
            self.netOuts.append(I)                                              # keeping track of the outputs of every layer
            
            #The input is propagated through the layers
            for idx in range(self.size):
                W = self.weights[idx]
                
                I = np.dot(I,W)                                                 #performs the dot product between the input vector and the weight matrix
                self.netIns.append(I)                                           # keeping track of the inputs to each layer
                
                #if we are on the last layer, we use the output activation function
                if idx == self.size -1:
                    I = self.outActiv_fun(I)
                #otherwise, we use the activation for the hidden layers
                else:
                    I = self.hiddenActiv_fun(I)
                    I = np.append(I,[1])
                    self.netOuts.append(I)
            
            self.out = I
            return self.out

        else:
            """implement batch training"""
            pass
        
    def reversed_feed(self, outIn):
        """
        similar to the feedforward function but reversed. It takes an output or target vector,
        and returns the corresponding input vector. Nothing is stored by this function.
        
        outIn: the target vector that will be the input for this function. It would be the output of the normal feedforward fucntion.
        
        """
        I = np.array(outIn)
        for W in self.weights[::-1]:                # We traverse backwards through the weight matrices
            I = np.dot(W,I)[:-1]                #The dot product of the two numpy arrays will have one extra element, corresponding to the bias node, and we do not need it, so we slice it off
        return I
            
                   
        
    def backprop(self,inputs,target, batch=False):
        """
        Backpropagation (online)
        inputs: a vector of inputs for the neural network. It corresponds to one training example (in online mode)
        target: a vector of expected values correspoding to the inputs vector
        batch: boolean flag. Indicates whether to use batch or online training. BATCH NOT IMPLEMENTED
        """
        #Gradients = [None]*self.size                        # it will have the same size as self.weights
        
        output = self.feedforward(inputs)                                       # performs forward propagation of the inputs 
        
        # Compute the error for the network at this particular example
        error = 0.5 * np.sum((target-output)**2)
        delta = None
        gradients = None 
        
        for i in range(self.size):
            back_index =self.size-1 -i                  # This will be used for the items to be accessed backwards            
            if i==0:
                # First, we calculate the delta for the output layer by taking the partial derivatives of the error function and more
                delta = (output-target) * self.outActiv_fun(self.netIns[back_index], derivative=True)
                gradients = np.outer(self.netOuts[back_index], delta)
                self.Gradients[back_index] = gradients

            else:
                # The calculation for the hidden deltas is slightly different than for the output neurons
                W = self.weights[back_index+1]                
                delta = np.dot(W,delta)[:-1] * self.hiddenActiv_fun(self.netIns[back_index], derivative=True)              #we slice off the delta value corresponding to the bias node
                #delta = np.dot(delta, W) * self.hiddenActiv_fun(self.netOuts[back_index], derivative=True)
                gradients = np.outer(self.netOuts[back_index], delta)           # the transpose is necessary to get a matrix of the correct shape. This can be avoided by changing the way the matrix is represented
                
                self.Gradients[back_index] = gradients
        
        
        if not batch:
            # when we want online training, weights are updated on the flight. 
            # otherwise, we just return the error
            # Update the weights on every training sample, because this is online training
            for i in range(self.size):
                delta_weight = self.learningRate * self.Gradients[i]
                self.weights[i] -= delta_weight + self.momentum*self.last_change[i]
                self.last_change[i] = self.Gradients[i]
        else:
            # This clause is for batch training.
            # We will iterate through the cumulative gradients 
            for k in range(self.size):
                self.batch_gradients[k] += self.Gradients[k]
                

            
        return error
    
        
    def trainEpoch(self,trainingSet, batch_switch = False):
        """
        Presents every training example to the network once, backpropagating the error
        for each one.
        trainingSet: a list of tuples pairing inputs,targets for each training example.
        Returns: cumulative error of the epoch
        """
        epoch_error = 0
        
        for inputs,targets in trainingSet:
            epoch_error += self.backprop(inputs,targets, batch_switch)
        
        if batch_switch is True:
            for i in range(self.size):
                self.batch_gradients[i] -= self.batch_gradients[i]              # we need the values to start at zero for every epoch
                delta_weight = self.learningRate * self.batch_gradients[i]
                self.weights[i] -= delta_weight + self.momentum*self.last_change[i]
                self.last_change[i] = self.batch_gradients[i]
        
        
        return epoch_error
        
    
    def train(self,trainingSet,epochs=1000,threshold_error = 1E-10, batch=False):
        """
        Trains the network for the specified number of epochs.
        trainingSet: a list of tuples pairing inputs,targets for each training example.
        epochs: The number of iterations of the training process. One epoch is completed when
                    all the training samples have been presented to the network once.
        threshold_error: The maximum error that the network should have. After completing one epoch,
                            if the error of the network is below the threshold, the training stops,
                            otherwise, it must keep going until the error is lower, or the specified number
                            of epochs has been reached.
        """
        self.last_change = [np.zeros(Mat.shape) for Mat in self.weights]
        
        if batch is True:
            # if we are using batch training, we create a list that will contain the accumulated gradient change for all the training patterns:
            self.batch_gradients = [np.zeros(Mat.shape) for Mat in self.last_change]

        for i in range(epochs+1):
            error = self.trainEpoch(trainingSet, batch)
            
            if i % (epochs/100) == 0:                                            # Every certain number of iterations, information about the network will be printed. Increase the denominator for more printing points, or reduce it to print less frequently
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
        
    
