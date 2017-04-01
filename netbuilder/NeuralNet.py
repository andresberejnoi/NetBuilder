# -*- coding: utf-8 -*-
"""
    @author: andresberejnoi
"""
#TODO: During training, if the network gets stuck in a local minima for several epochs,
# then randomly modify certain weights in the matrix. This might allow the network to get out
# of that minima and converge 

import numpy as np
import _param_keys as keys
from activations import *
#import tools         # this is a python file where I will put some functions before I decide to include them here directly

    
    
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

class Network(object):
    """
    Implements the methods and attributes of an Artificial Neural Network (Multilayer).
    The network implemented is a feedforward one, with backpropagation as the training algorithm.
    This implementation uses the numpy library, so it needs to be available to be able to run
    """
    #np.random.seed()                       # start the random seed before using it
    #np.random.random()
    #Set up a dictionary of activation functions to access them more easily

    def __init__(self):
        self.topology = None
        self.learningRate = None
        self.momentum = None
        self.name = None 
        self._hiddenActiv_fun_key = None
        self._outActiv_fun_key = None
        self.output_activation = None
        self.hidden_activation = None
    
    def init(self,topology,learningRate=0.01,momentum=0.1,name='Network',add_bias=True):
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
        self.topology = topology
        self.learningRate = learningRate
        self.momentum = momentum
        self.name = name
        self.size = len(self.topology)-1           #The size of the network will be the number of weeight matrices between layers, instead of the number of layers itself
        #self._hiddenActiv_fun_key = 'tanh'
        #self._outActiv_fun_key = 'tanh'
        #self.output_activation = self.set_outActivation_fun(func=self._outActiv_fun_key)
        #self.hidden_activation = self.set_hiddenactivation_fun(func=self._hiddenActiv_fun_key)
        
        # Initialize random weights, and create empty matrices to store the previous changes in weight (for momentum):
        if add_bias:
            #self.weights = [np.random.normal(loc=0,scale=0.6,size=(topology[i]+1, topology[i+1])) for i in range(self.size)]              
            self.weights = [np.random.normal(loc=0,
                                             scale=0.6,
                                             size=(topology[i]+1, topology[i+1]+1)) for i in range(self.size-1)] #we are only generating matrices for inital and hidden layers
            #Create matrix for output layer
            f_idx = self.size-1     #use this index for the final layer matrix below
            self.weights.append(np.random.normal(loc=0,
                                                 scale=0.6,
                                                 size=(topology[f_idx]+1,topology[f_idx+1])))
        else:
            raise NotImplemented("Currently the network only works when bias nodes are used")
            self.weights = [np.random.normal(loc=0,scale=0.6,size=(topology[i], topology[i])) for i in range(self.size)]

        self.Gradients = [None] * self.size

    # Initializer helpers
    def _init_from_file(self,params,weights_dict):
        self.name = params[keys._nane]
        self.topology = params[keys._topology]
        self.learningRate = params[keys._learning_rate]
        self.momentum = params[keys._momentum]
        #self._outActiv_fun_key = params[keys._output_activation]
        #self._hiddenActiv_fun_key = params[keys._hidden_activation]
        #self.output_activation = self.set_outActivation_fun(func=self._outActiv_fun_key)
        #self.hidden_activation = self.set_hiddenactivation_fun(func=self._hiddenActiv_fun_key)
        
        #unpack weights
        self.weights = [weights_dict[layer_mat] for layer_mat in weights_dict]
        self.size = len(self.weights)
        self.Gradients = [None]*self.size
    
    #--------------------------------------------------------------------------
    # Overloading Operators:
    #
    def __str__(self):
        '''
        For now, the string method simply returns the topology of the network.
        '''
        return "Network: {0}".format(self.topology)
    
    __repr__ = __str__


    #---------------------------------------------------------------------------
    # Getters
    #
    def get_complexity(self):
        """
        Returns the number of features or synapses (connections) present in the network.
        """
        synapses = 0
        for mat in self.weights:
            synapses += mat.size
        return synapses

    def get_num_nodes(self):
        """Returns the number of nodes in the network (includes input and output nodes)"""
        return sum(self.topology)

    def get_connection_layer(self, idx):
        """
        idx: int; the index corresponding to a layer in self.weights.
        returns: The connection weights for the layer requested (self.weights[idx])
        """
        try:
            return self.weights[idx]
        except:
            print("""Could not find layer {0} in network.\nNetwork has {1} layers.""".format(idx, self.size))
    
    def _get_model(self):
        """
        Returns a dictionary of network parameters that can be used for a configuration file. This function is used when saving the network.
        """
        parameters = {keys._topology:self.topology,
                      keys._size:self.size,
                      keys._name:self.name,
                      #keys._output_activation:self._outActiv_fun_key,
                      #keys._hidden_activation:self._hiddenActiv_fun_key,
                      keys._learning_rate:self.learningRate,
                      keys._momentum:self.momentum}
        
        return parameters
    #--------------------------------------------------------------------------
    # Section below is for setters
    #
    
    #
    # Functionality of the network
    #              
                
    def feedforward(self,inputs,hidden_activation=tanh,output_activation=tanh):
        """
        Performs the feedforward propagation of the inputs through the layers.
        inputs: numpy array of shape [number of samples x number of features per sample]; inputs to the first layer
        """
        # These two lists will contain the inputs and the outputs for each layer, respectively
        self.netIns = []                                                        
        self.netOuts = []
        
        input_samples=inputs.shape[0]
        
        #Currently, this will cause a crash when the network was created without bias nodes
        I = np.concatenate((inputs,np.ones((input_samples,1))),axis=1)                # adds the bias input of 1
        self.netOuts.append(I)                                              # keeping track of the outputs of every layer
        
        #The input is propagated through the layers
        for idx in range(self.size):
            W = self.weights[idx]
            
            I = np.dot(I,W)                                                 #performs the dot product between the input vector and the weight matrix
            self.netIns.append(I)                                           # keeping track of the inputs to each layer
            
            #if we are on the last layer, we use the output activation function
            if idx == self.size -1:
                I = output_activation(I)
            #otherwise, we use the activation for the hidden layers
            else:
                I = hidden_activation(I)
                #I = np.concatenate((I,np.ones((I.shape[0],1))), axis=1)
                self.netOuts.append(I)
        
        #self.out = I
        return I

        
    def reversed_feed(self, outIn):
        """
        similar to the feedforward function but reversed. It takes an output or target vector,
        and returns the corresponding input vector. Nothing is stored by this function.
        
        outIn: the target vector that will be the input for this function. It would be the output of the normal feedforward fucntion.
        
        """
        I = np.array(outIn)
        for W in self.weights[::-1]:                # We traverse backwards through the weight matrices
            I = np.dot(W,I)[:-1]                #The dot product of the two numpy arrays will have one extra element, corresponding to the bias node, but we do not need it, so we slice it off
        return I
    
    def _compute_error(self,expected_out,actual_out,error_func):
        """
        Computes the error for the network output.
        expected_out: numpy array with shape [batch_size (or number of samples) x number of output features]; 
                        This array should contain the target values that we want the network to produce once trained.
        actual_out: numpy array with shape equal to expected_out
        error_func: a function to compute the error
        """
        error = error_func(expected_out,actual_out)
        return error
    
    def optimize(self,gradients):
        """
        gradients: iterable containing numpy arrays; each numpy array is the gradient matrix computed by backpropagation for each layer matrix
        """
        for k in range(self.size):
            delta_weight = self.learningRate * gradients[k]
            full_change = delta_weight + self.momentum*self.last_change[k]
            self.weights[k] -= full_change
            self.last_change[k] = 1*gradients[k] #copy gradient mat
        
                
    def backprop(self, input_samples,target,output, error_func, batch_mode=True,hidden_activation=tanh,output_activation=tanh):
        """
        Backpropagation
        input_samples: numpy array of all samples in a batch
        target_outputs: numpy array of matching targets for each sample
        output: numpy array; actual output from feedforward propagation. It will be used to train the network
        batch_mode: boolean flag. Indicates whether to use batch or online training.
        error_func: function object; this is the function that computes the error of the epoch and used during backpropagation.
                    It must accept parameters as: error_func(target={target numpy array},actual={actual output from network},derivative={boolean to indicate the operation mode})
        """
        #Define placeholder variables
        delta = None
        gradient_mat = None
        
        #Compute gradients and deltas
        for i in range(self.size):
            back_index =self.size-1 -i                  # This will be used for the items to be accessed backwards  
            if i!=0:
                W_trans = self.weights[back_index+1].T        #we use the transpose of the weights in the current layer
                d_activ = hidden_activation(self.netIns[back_index],derivative=True)
                d_error = np.dot(delta, W_trans)
                delta = d_error * d_activ
                gradient_mat = np.dot(self.netOuts[back_index].T , delta)
                self.Gradients[back_index] = gradient_mat
            else:
                #Herewe calculate gradients for final layer
                d_activ = output_activation(self.netIns[back_index],derivative=True)
                d_error = error_func(target,output,derivative=True)
                delta = d_error * d_activ
                gradient_mat = np.dot(self.netOuts[back_index].T , delta)
                self.Gradients[back_index] = gradient_mat
        # Update weights using the computed gradients
        for k in range(self.size):
            delta_weight = self.learningRate * self.Gradients[k]
            full_change = delta_weight + self.momentum*self.last_change[k]
            self.weights[k] -= full_change
            self.last_change[k] = 1*self.Gradients[k]
       
    def TrainEpochOnline(self,input_set,target_set):
        """
        Presents every training example to the network once, backpropagating the error
        for each one.
        trainingSet: a list of tuples pairing inputs,targets for each training example.
        Returns: cumulative error of the epoch
        """
        epoch_error = 0
        
        for i in range(len(input_set)):
            epoch_error += self.backprop_old(input_set,target_set)
                
        return epoch_error
        
    def train(self,input_set,
              target_set,
              epochs=5000,
              threshold_error=1E-10,
              batch_mode=True,
              batch_size=0,
              error_func=mean_squared_error,
              hidden_activation=tanh,
              output_activation=tanh,
              print_rate=100):
        """
        Trains the network for the specified number of epochs.
        input_set: numpy array of shape [number of samples x number of features per sample]
        target_set: numpy array of shape [number of samples x number of features per output]
        epochs: The number of iterations of the training process. One epoch is completed when
                    all the training samples in a batch have been presented to the network once.
        threshold_error: The maximum error that the network should have. After completing one epoch,
                            if the error of the network is below the threshold, the training stops,
                            otherwise, it must keep going until the error is lower, or the specified number
                            of epochs has been reached.
        batch_mode: boolean flag; tells the program whether to do batch or online training (True is for batch)
        batch_size: int; how many samples will make one mini batch. It is 0 by default, which means that one batch will contain all samples.
        error_func: function object; this is the function that computes the error of the epoch and used during backpropagation.
                    It must accept parameters as: error_func(target={target numpy array},actual={actual output from network},derivative={boolean to indicate the operation mode})
        print_rate: int: controls the frequency of printing. It tells the trainer to print the error every certain number of epochs: print if current epoch is a multiple of print_rate.
                        Increase this number to print less often, or reduce it to print more often.
        """
        #initialize placeholders:
        self.last_change = [np.zeros(Mat.shape) for Mat in self.weights]
        self.batch_gradients = [np.zeros(Mat.shape) for Mat in self.weights]
    
        #Check if it should do batch training
        if batch_mode:
            if batch_size > 0:
                num_samples = input_set.shape[0]
                try:
                    assert(batch_size <= num_samples)
                except AssertionError:
                    print ("""Batch size '{0}' is bigger than number of samples available: '{1}'""".format(batch_size,num_samples))
                    raise
                
                #Define number of iterations per epoch:
                num_iterations = num_samples // batch_size + (1 if num_samples%batch_size > 0 else 0)
                
                for epoch in range(epochs+1):
                    #define start and end index through the data
                    start_idx = 0
                    end_idx = batch_size
                    error = 0
                    for i in range(num_iterations):                    
                        #Prepare mini batch
                        mini_inputs = input_set[start_idx:end_idx]
                        mini_targets = target_set[start_idx:end_idx]
                        
                        #Feed Network with inputs to compute error
                        output = self.feedforward(mini_inputs,hidden_activation=hidden_activation,output_activation=output_activation)
                        error += error_func(target=mini_targets,actual=output)
                        #print('Error:',error,'Epoch:',epoch,'iter:',i)
                        #compute the error
                        self.backprop(input_samples=mini_inputs,
                                              target=mini_targets,
                                              error_func=error_func,
                                              hidden_activation=hidden_activation,
                                              output_activation=output_activation,
                                              output=output)
                        
                        #TODO: Read the mini batch data from some file or generator. The current implementation loads the whole batch in memory and then
                        # takes mini batches from there, but this makes the mini batch method pointless (sort of)
                        
                        #Update indexes
                        if end_idx < num_samples:       #increase indexes while there is more data 
                            start_idx = end_idx
                            if (num_samples-end_idx) < batch_size:
                                end_idx = num_samples
                            else:
                                end_idx += batch_size
                        #else:
                        #    raise NetworkError("""End index for mini batches went out of range: end index:{0} / number of samples:{1}""".format(end_idx,num_samples))
                        
                    #print information about training
                    if epoch % print_rate == 0:                                            # Every certain number of iterations, information about the network will be printed. Increase the denominator for more printing points, or reduce it to print less frequently
                        self.print_training_state(epoch,error)
                    if error <= threshold_error:                                        # when the error of the network is less than the threshold, the traning can stop
                        self.print_training_state(epoch,error, finished=True)
                        break
                    
                    
            else:
                mini_inputs = input_set
                mini_targets = target_set
                for epoch in range(epochs+1):
                    #Feed Network with inputs can compute error
                    output = self.feedforward(mini_inputs,hidden_activation=hidden_activation,output_activation=output_activation)
                    error = error_func(target=mini_targets,actual=output)
                    #print('Error:',error,'Epoch:',i)
                    
                    #compute the error
                    self.backprop(input_samples=mini_inputs,
                                          target=mini_targets,
                                          error_func=error_func,
                                          hidden_activation=hidden_activation,
                                          output_activation=output_activation,
                                          output=output)
                    
                    #if epoch % (epochs/print_rate) == 0:                                            # Every certain number of iterations, information about the network will be printed. Increase the denominator for more printing points, or reduce it to print less frequently
                    if epoch % print_rate == 0:
                        self.print_training_state(epoch,error)
                    if error <= threshold_error:                                        # when the error of the network is less than the threshold, the traning can stop
                        self.print_training_state(epoch,error, finished=True)
                        break
        else:
            raise NotImplementedError
            for epoch in range(epochs+1):
                #compute error
                error = self.TrainEpochOnline(input_set=input_set,
                                              target_set=target_set)
                #print information about training
                if epoch % print_rate == 0:                                            # Every certain number of iterations, information about the network will be printed. Increase the denominator for more printing points, or reduce it to print less frequently
                    self.print_training_state(epoch,error)
                if error <= threshold_error:                                        # when the error of the network is less than the threshold, the traning can stop
                    self.print_training_state(epoch,error, finished=True)
        
    # Information printers
    def print_training_state(self,epoch,error,finished=False):
        """Prints the current state of the training process, such as the epoch, current error"""
        #print("Epoch:",iterCount)
        if finished:
            print("Network has reached a state of minimum error.")
        #print("Error: {0}\tEpoch {1}".format(error,iterCount))
        print("""Epoch {0} completed""".format(epoch),'Error:',error)
    
    def _cleanup(self):
        """
        Sets containers back to their original state. It is a test function for now.
        """
        self.netIns = []
        self.netOuts = []
        self.Gradients = [None]*self.size
        
################################################################################################