import math
#import numpy as np
#class Network (object):
#    layerCount = 0
#    shape = None
#    weights = []
#    
#    def __init__(self, layersize):
#        self.layerCount = len(layersize) -1
#        self.shape = layersize
#        
#        self._layerInput = []
#        self._layerOput = []
#        
#        for (l1, l2) in zip(layersize[:-1], layersize[1:]):
#            self.weights.append(np.random.normal(scale = 0.01, size = (l2, l1+1)))



#def activation(inputs, weights):
#    '''
#    inputs: An array of the input values
#    weights: An array of the weights'''
#    activation = 0.0
#    for i in range(len(inputs)):
#        activation += inputs[i]*weights[i]
#    return activation


class Perceptron(object):
    '''A perceptron class. It is a single-layer neural network with binary output.'''
    def __init__(self, array_weights, threshold=1.0 ):
        '''Initializes the perceptron'''
       # self.inputs = array_inputs
        self.weights = array_weights
        self.threshold = float(threshold)
        self.bias = -1*self.threshold
        self.output = None
        
    def activationFunction(self, array_inputs):
        '''Determines the output of the perceptron based on the inputs 
        and the respective weights'''
        value = sum([W*I for W,I in zip(self.weights, array_inputs)])        #Sums the prouct of the inputs with the weights
        self._getOutput(value)
        return self.output
    
    def _getOutput(self, value):
        '''Returns the output value.
        "It might not be necessary to make in a separate function".'''
        if value + self.bias >= 0:              #The concept of the bias was followed from here: http://neuralnetworksanddeeplearning.com/chap1.html
            self.output = 1
        else:
            self.output = 0
        
    def get_config(self):
        '''Returns relevant information about the perceptron in the form of a tuple'''
        return (self.weights, self.threshold)


class NeuralNetwork(object):
    '''Provides methods for creating and manipulating an artificial neural network.
    The network should be trained using the backpropagation method.'''
    def __init__(self, topology, array_weights):
        '''Initializes all the instance variables in the network.
        topology: a list of integers; each integer refers to the number of neurons
                    in a given layer. It goes from left to rights, [input_layer, hidden_layer,...,output_layer]'''
        self.topology = topology
        self.weights = array_weights
        self.layers = None                      #an array of the layers in the network. Should have the same order as in self.topology
        
  
        
        
class Layer(object):
    '''Provides the methods for instantiating a layer for a neural network'''
    
    def __init__(self, numNeurons, array_weights, array_errors = None):
        self.numNeurons = numNeurons
        self.weights = array_weights
        self.errors = array_errors
        
class Neuron(object):
    '''A single neuron in a multi-layer network.'''
    def __init__(self, array_weights, type, learning_rate = 0.01):
        '''Initializes the neuron.
        array_weights: an array of the weights from the previous layer to this neuron.
        type: a string of one character. It represents the type of neuron: "I" for input, "H" for hidden, and "O" for output.
        learning_rate = a float'''
        self.weights = array_weights
        self.output = None
        self.bias = 1
        self.type = type.upper()                    #the type will always be an uppercase letter.
        self.Lr = round(float(learning_rate), 2)
        self.errorSignal = None
#        self.inputs = None

        
    def _net_input(self, inputs):
        '''Calculates the net input to the current neuron.'''
        input = sum([W*I for W,I in zip(self.weights, inputs)]) + self.bias
        return input
        
    def sigmoidActivation(self, inputs):
        '''Calculates the output of the neuron using a sigmoid function.
        inputs: an array of input values from the previous layer.'''
        self.output = 1/(1+ math.e**(-1*self._net_input(inputs)))
        return self.output
    def error(self, true_output):
        '''Calculates the error signal of the current neuron.'''
        if self.type == "O":                #If the neuron is in the output layer
            self.errorSignal = (true_output-self.output) * self.output*(1-self.output)
        elif self.type =="H":
            self.errorSignal = ()               #FIX THIS!!!!!!!!!!!!!!!
    def updateWeights(self, inputs):
        '''Updates the weights to the current neuron from the previous layer'''
        for i in range(len(self.weights)):
            self.weights[i] = self.Lr*self.sigmoidActivation(inputs)*self.error()
        
    def get_type(self):
        '''Returns the type of the neuron.'''
        return self.type
        
    def is_outputNode(self):
        '''Returns True if the node is an output node. Returns False otherwise.'''
        
        
        
