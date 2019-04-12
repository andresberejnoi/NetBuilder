# -*- coding: utf-8 -*-
"""@author: andresberejnoi

TODO: During training, if the network gets stuck in a local minima for several epochs,
then randomly modify certain weights in the matrix. This might allow the network to get out
of that minima and converge.

"""

#import numpy as np
from netbuilder import np
from . import _param_keys as keys #import keys for saving and loading network from file
from .activations import *  #import activation functions defined in the file, such as tanh and sigmoid
from .loss import *
#import tools         # this is a python file where I will put some functions before I decide to include them here directly

#---------------------------------------------------------------------------------------------
class NetworkError(Exception):
    """ An exception object that can be raised to handle different situations.

    It is currently very simple.

    """
    def __init__(self, msg):
        """Sets value for error message.

        Parameters
        ----------
        msg : string
            Message to be displayed when this exception is raised.

        """

        self.msg = msg

    def __str__(self):
        return self.msg

#----------------------------------------------------------------------------------------------

class Network(object):
    """Implements the methods and attributes of an Artificial Neural Network (Multilayer).

    The network implemented is a feedforward one, with backpropagation as the training algorithm.
    This implementation uses the numpy library, so it needs to be available to be able to run.

    """
    #np.random.seed()                       # start the random seed before using it
    #np.random.random()
    #Set up a dictionary of activation functions to access them more easily

    def __init__(self):
        """Setting names for instance variables.

        """
        self.topology = None
        self.learningRate = None
        self.momentum = None
        self.name = None
        self.size = None
        #self._hiddenActiv_fun_key = None
        #self._outActiv_fun_key = None
        #self.output_activation = None
        #self.hidden_activation = None

    #----------------------------------------------------------------------------------
    # Initializers
    #
    def init(self,topology,learningRate=0.01,momentum=0.1,name='Network',add_bias=True):
        """Initializes the network with specified shape and parameters.

        Parameters
        ----------
        topology : python iterable
            A Python list with integers indicating the shape of the network.
            i.e: [5,10,1]: this encodes a network of 3 layers (one input, 1 hidden, and 1 output).
            The input layer will have 5 neurons, the hidden layer will have 10, and the output layer will have only one neuron.
        learningRate : float, optional
            It helps with the speed and convergence of the network. It is usually small.
            A very small number will cause the network to converge very slowly. A high rate will make
            the network oscillate during training and prevent it from "learning" patterns.
        momentum : float, optional
            It is also used during the training process. It is related to how much the previous changes
            affect the new ones.
        name : string, optional
            A name to identify the network more easily if it is saved to a file.

        """
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

    def _init_from_file(self,params,weights_dict):
        """Initializes network with values saved on files.

        Parameters
        ----------
        params : dictionary
            Maps valid keys that can be obtained from reading a configuration file (or using _get_model()) with parameters that the network needs.
        weights_dict : dictionary
            The dictionary generated from reading an npz numpy file with numpy.load.

        """

        self.name = params[keys._name]
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

    #-------------------------------------------------------------------------------------------
    # Overloading Operators:
    #
    def __str__(self):
        """For now, the string method simply returns the topology of the network.

        """

        return "Network: {0}".format(self.topology)

    __repr__ = __str__


    #---------------------------------------------------------------------------
    # Getters
    #
    def get_num_connections(self):
        """Returns the number of features or synapses (connections) present in the network.

        """

        synapses = 0
        for mat in self.weights:
            synapses += mat.size
        return synapses

    def get_num_nodes(self):
        """Returns the number of nodes in the network (includes input and output nodes).

        """

        return sum(self.topology)

    def get_connection_mat(self, idx):
        """Gets the matrix weight at position idx in self.weights.

        Parameters
        ----------
        idx : int
            Index corresponding to a layer in self.weights.

        Returns
        -------
        numpy array
            The connection weights for the layer requested (self.weights[idx])

        """

        try:
            return self.weights[idx]
        except:
            print("""Could not find layer {0} in network.\nNetwork has {1} layers.""".format(idx, self.size))

    def _get_model(self):
        """Returns a dictionary of network parameters that can be used for a configuration file. This function is used when saving the network.

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
    # Functionality of the network
    #

    def feedforward(self,inputs,hidden_activation=tanh,output_activation=tanh):
        """Performs the feedforward propagation of the inputs through the layers.

        Parameters
        ----------
        inputs : numpy array
            Shape of array should be [number of samples x number of features per sample].
            This array contains inputs to the first layer.
        hidden_activation : function object, optional
            It is the activation function for hidden layers. It must be able to accept numpy arrays.
        output_activation : function object, optional
            It is the activation function for final layer. It must be able to accept numpy arrays.

        Returns
        -------
        numpy array
            Shape will be [number of samples x number of output features].

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

    def predict(self,inputs,hidden_activation=tanh,output_activation=tanh):
        """This function is very similar to feedforward, but makes sure that the input is in the correct format. It is intended for testing the final network without adding additional if statements into the feedfoward function which will be used during training.

        Parameters
        ----------
        inputs : numpy array
            Shape of array should be [number of samples x number of features per sample].
            Inputs to the first layer.
        hidden_activation : function object, optional
            It is the activation function for hidden layers. It must be able to accept numpy arrays.
        output_activation : function object, optional
            It is the activation function for final layer. It must be able to accept numpy arrays.

        Returns
        -------
        numpy array
            Shape will be [number of samples x number of output features].

        """

        I = inputs
        #if the input is a list and not a numpy array:
        if not isinstance(I,np.ndarray):   #if imput is not numpy array
            I = np.array(I)

        #now we arrange the inputs to be organized in rows if it is only one column.
        #for example, if we have an array: array ([0,1,2,3,4,5,6,7,8,9]), its shape will be (10,) but we need it to be (1,10) as: array ([[0,1,2,3,4,5,6,7,8,9]])
        if len(I.shape) == 1:
            I = I.reshape((1,I.shape[0]))

        output = self.feedforward(I,hidden_activation=hidden_activation,output_activation=output_activation)
        return output


    def reversed_feed(self, outIn):
        """ Like the feedforward function but reversed. It takes an output or target vector, and returns the corresponding input vector. Nothing is stored by this function.

        Parameters
        ----------
        outIn : numpy array
            The target vector that will be the input for this function. It would be the output of the normal feedforward fucntion.

        Returns
        -------
        numpy array
            output of running the network backwards.

        """

        I = np.array(outIn)
        for W in self.weights[::-1]:                # We traverse backwards through the weight matrices
            I = np.dot(W,I)[:-1]                #The dot product of the two numpy arrays will have one extra element, corresponding to the bias node, but we do not need it, so we slice it off
        return I

    def _compute_error(self,expected_out,actual_out,error_func):
        """Computes the error for the network output.

        Parameters
        ----------
        expected_out : numpy array
            Shape of array should be [batch_size (or number of samples) x number of output features].
            This array should contain the target values that we want the network to produce once trained.
        actual_out : numpy array
            Shape should be equal to `expected_out`
        error_func : function object
            A function to compute the error (difference between the two inputs).

        Returns
        -------
        float
            The sum of all the differences between the two inputs.

        """

        error = error_func(expected_out,actual_out)
        return error

    def optimize(self,gradients):
        """Uses the gradients computed by the backpropagation method to update network weights.

        performs stochastic gradient descent and adjusts the weights


        Parameters
        ----------
        gradients : python iterable
            This iterable {list, tuple, etc.} contains numpy arrays.
            Each numpy array is the gradient matrix computed by backpropagation for each layer matrix.

        """

        for k in range(self.size):
            delta_weight = self.learningRate * gradients[k]
            full_change = delta_weight + self.momentum*self.last_change[k]
            self.weights[k] -= full_change
            self.last_change[k] = 1*gradients[k] #copy gradient mat


    def backprop(self, input_samples,target,output, error_func, hidden_activation=tanh,output_activation=tanh):
        """Backpropagation.

        Parameters
        ----------
        input_samples : numpy array
            Contains all samples in a batch.
        target_outputs : numpy array
            Matching targets for each sample in `input_samples`.
        output : numpy array
            Actual output from feedforward propagation. It will be used to check the network's error.
        batch_mode : bool, Don't use for now.
            Indicates whether to use batch or online training.
        error_func : function object
            This is the function that computes the error of the epoch and used during backpropagation.
            It must accept parameters as: error_func(target={target numpy array},actual={actual output from network},derivative={boolean to indicate operation mode})
        hidden_activation : function object, optional
            It is the activation function for hidden layers. It must be able to accept numpy arrays.
            It must also provide a parameter to indicate operation in derivative or normal mode.
        output_activation : function object, optional
            It is the activation function for final layer. It must be able to accept numpy arrays.
            It must also provide a parameter to indicate operation in derivative or normal mode.

        """

        #placeholder variables
        #delta = None
        #gradient_mat = None

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
        self.optimize(gradients=self.Gradients)

    def train(self,input_set,
              target_set,
              epochs=5000,
              threshold_error=1E-10,
              batch_size=0,
              error_func=mean_squared_error,
              hidden_activation=tanh,
              output_activation=tanh,
              print_rate=100):
        """Trains the network for the specified number of epochs.

        Parameters
        ----------
        input_set : numpy array
            Shape of array should be [number of samples x number of features per sample].
        target_set : numpy array
            Shape of array should be [number of samples x number of features per output].
        epochs : int, optional
            The number of iterations of the training process. One epoch is completed when
            all the training samples in a batch have been presented to the network once.
        threshold_error : float, optional
            The maximum error that the network should have. After completing one epoch,
            if the error of the network is below `threshold_error`, the training stops,
            otherwise, it must keep going until the error is lower, or the specified number
            of epochs has been reached.
        batch_size : int, optional
            How many samples will make one mini batch. It is 0 by default, which means that one batch will contain all samples. Set to 1 for online training.
        error_func : function object, optional
            This is the function that computes the error of the epoch and used during backpropagation.
            It must accept parameters as: error_func(target={target numpy array},actual={actual output from network},derivative={boolean to indicate the operation mode})
        print_rate : int, optional
            Controls the frequency of printing. It tells the trainer to print the error every certain number of epochs: print if current epoch is a multiple of print_rate.
            Increase this number to print less often, or reduce it to print more often.

        """

        #TODO: Read the mini batch data from some file or generator. The current implementation loads the whole batch in memory and then
        # takes mini batches from there, but this makes the mini batch method pointless (sort of)

        #initialize placeholders:
        self.last_change = [np.zeros(Mat.shape) for Mat in self.weights]

        #Check if it should do batch training, mini batch, or full batch
        num_samples = input_set.shape[0]

        try:    #check that batch_size makes sense
                assert(batch_size <= num_samples)
        except AssertionError:
                print ("""Batch size '{0}' is bigger than number of samples available: '{1}'""".format(batch_size,num_samples))
                raise
        #----------------------------------------------------------------------
        if 0 < batch_size < num_samples:    #here we do mini batch or online

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

        else:       #here we do full batch
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

    # Information printers
    def print_training_state(self,epoch,error,finished=False):
        """Prints the current state of the training process, such as the epoch, current error.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        error: float
            Network error for current epoch.
        finished : bool, optional
            If true, then a message indicating training is complete is printed. Otherwise
            just print epoch and error normally.

        """

        #print("Epoch:",iterCount)
        if finished:
            print("Network has reached a state of minimum error.")
        #print("Error: {0}\tEpoch {1}".format(error,iterCount))
        print("""Epoch {0} completed""".format(epoch),'Error:',error)

    def _cleanup(self):
        """Sets containers back to their original state. It is a test function for now.

        """

        self.netIns = []
        self.netOuts = []
        self.Gradients = [None]*self.size

################################################################################################
