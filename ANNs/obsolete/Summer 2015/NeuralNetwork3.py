import math
import random
import matrix
#import numpy as np

class NetworkError(Exception):
    '''Still in development. It should provide useful feedback as to why an error occurred.'''
    def __init__(self, message = "An error with the network has occurred."):
        self.message = message
    def __str__(self):
        return self.message
    def __repr__(self):
        return str(self)

class Network(object):

    def __init__(self, topology, learning_rate = 0.01):

        self.topology = topology
        self.Lr = learning_rate
        self.netOutput = None
        self.net = []
        self.numNeurons = sum(topology)
        self.connections = []

        self._expectedOut = None

    #Generating the network with random weights:
        self._generateNetwork()
        self.initRandWeights()

    def _generateNetwork(self):
        idx = 0
        layer_index = 0
        for layer in self.topology:
            self.net.append([])                 #appends a new empty layer to the network
            for i in range(layer):
                node = Neuron()
                node.index = i                  #If the index relative to only the layer is needed, instead of idx, use i
                if layer_index == 0:
                    node.setType('I')
                elif layer_index == len(self.topology)-1:
                    node.setType('O')
                else:
                    node.setType('H')
                self.net[layer_index].append(node)
                node.links = 0
                idx += 1
            layer_index += 1

    def initRandWeights(self):
        weights = []
        up = 0.5                                                                #up is the upper boundary for the random float numbers for the weight
        down = -0.5                                                             #down is the lower boundary
        idx = 0
        inverse = self.net[::-1]
        for layer in inverse[:len(inverse)-1]:               #This skips the input layer, which will not be considered units in the network
            weights.append((matrix.randFloatMatrix(len(layer), len(inverse[idx+1]), down, up), matrix.randFloatMatrix(len(layer),1, down, up)))   #This Might be the problem: len(layer) is not what you think it is, maybe.
            idx += 1
        self.connections = weights

#####    def _createConnections(self):
#####        '''This function might not be needed anymore, or maybe later. We'll see...'''
######
######        for layer in self.net[::-1]:
######            for node in layer:
######                for i in range()
#####
#####    #Creating forward links starting from the input layer:
#####        idx = 0
#####        for layer in self.net:
#####            if idx == len(self.net)-1:                      #When we are at the last layer, we do not need to worry
#####                break
#####            for node in layer:
#####             #   for i in range(len(self.net[idx+1])):
#####                node.forwardLinks = [self.net[idx+1][i] for i in range(len(self.net[idx+1]))]
#####            idx += 1
#overloading
    def __len__(self):
        return self.numNeurons
    def __repr__(self):
        return str(self.topology)

#Learning functions and helpers:
    def _layerOutput(self, index):
        output = []
        for node in self.net[index]:
            if node.type=='O':
                output.append(node.outputActivation())
            else:
                output.append(node.activation())
        M = matrix.Matrix(len(output), 1, output)
        return M
        
    def _matrix_to_array(self):
        '''Takes the output value for the network and returns an array of it, stripping
        out the Matrix class.'''
        array_output = self.netOutput.get_col(0)
        return array_output

    def feedForward(self, inputs):
        '''Computes the final output for the network, based on the current weights
        and the provided inputs. It returns a Matrix object with the final results.'''

        I = matrix.Matrix(len(inputs), 1, inputs)                               #Creates a matrix object representing the input vector that is going to be fed into the network
        idx_layer = 1

        #The following for loop computes the activations for the hidden layers
        for layer,bias in self.connections[::-1]:
            for i in range(layer.row_len()):
                self.net[idx_layer][i].input = sum((layer[i]*I + bias[i]).matrix[0])         #calculates the input for each neuron in the next layer, based on the current inputs
            I = self._layerOutput(idx_layer)
            idx_layer += 1

        #The following loop computes the output for the ouptut layer
        self.netOutput = I                                                      #The output for the network is a Matrix object
        
        output_array = self._matrix_to_array()
        
        return output_array

#Backpropagation steps and helpers:
    def _derivative_tanh(self):
        ''''''
    def _getErrorLayer(self,index):
        '''The index goes from the output layer to input layer.
        Returns a list of the errors in the layer'''
        errors = []
        for node in self.net[::-1][index]:
            errors.append(node.error)

  #      E = matrix.Matrix(len)
        return errors


    def _getOutputSignals(self):
        '''Computes the error signals for the nodes in the output layer of the network.'''
        deltaOutput = self._expectedOut + self.netOutput*(-1)
        derivatives = []
        for node in self.net[::-1][0]:     #gets the nodes in the output layer
            derivatives.append(node._derivativeLinearOut)

        errorVector = []
        for i in range(deltaOutput.row_len()):
            _error = deltaOutput.get_row(i)[0]*derivatives[i]
            self.net[::-1][0][i].error = _error
#            errorVector.append(deltaOutput.get_row(i)[0]*derivatives[i])
            errorVector.append(_error)

        E = matrix.Matrix(len(errorVector),1,errorVector)
        return E

 #       matrix,bias = self.connections[0]                                       #The index 0 gets the output matrix and its respective bias matrix

    def backpropagate(self, real_outputs):
        '''Backpropagates the error through the layers
        1. Calculate the error signal for the output layer.
        2.  a) in each hidden layer, calculate each node's error signal
            b) Update the weights in the network. Use indexes, but with caution; May the force be with you...

        Additional notes: The final objective here is to modify the matrices stored in self.connections.
                    Also, remember that self.connections stores matrices in order [O,H....H,I], so the
                    first matrix is always the weights from the output to the next hidden layer.'''
        self._expectedOut =matrix.Matrix(len(real_outputs),1,real_outputs)      #Creates a matrix with the real output
        E = self._getOutputSignals()                                             #E is a matrix of order (outputs x 1). It contains the errors of the output layer
        index_layer = 1
        for weights_bias,layer in zip(self.connections,self.net[::-1]):
            weights,bias = weights_bias
            idx = 0
            for col_num in range(weights.col_len()):
                weights.matrix[idx][col_num] += self.Lr*E.matrix[idx][0] * layer[idx].input
                idx += 1
                if idx >= weights.row_len():
                    idx = 0
            #Before moving the next matrix of connections, find the error signals for the next hidden layer.
            E = self._getHiddenSignals(index_layer)
            index_layer += 1


    def _getHiddenSignals(self,layer_index):
        '''The hidden signal for a unit is error = previous_error*updated_weight*derivative_activation'''
        layer = self.net[::-1][layer_index]
        weights,bias = self.connections[layer_index-1]
        errors = self._getErrorLayer(layer_index-1)
        for node in layer:
            node.error = -1*sum([W*E for W,E in zip(weights.get_col(node.index),errors)])

        hiddenErrors = self._getErrorLayer(layer_index)

        ErrorMatrix = matrix.Matrix(len(hiddenErrors),1,hiddenErrors)

        return ErrorMatrix



class Neuron(object):

    def __init__(self):
        self.output = None
        self.input = None
        self.forwardlinks = None
        self.backLinks = None
        self.type = None
        self.error = None
        self.index = 0
        self.layer_pos = 0
        self._derivativeLinearOut = 5.0/2.0                                     #The derivative of the linear function. It is just a constant

    def _derivativeHiddenActivation(self):
        '''Returns the derivative of the tanh function used for the activation of
        the hidden units'''
        derivative = 1 - self.activation()**2
        return derivative

    def outputActivation(self):
        '''Computes the output of the neuron using a sigmoid function'''
        self.output = 1/(1+math.e**(-1*self.input))
        return self.output

    def activation(self):
        '''Uses the hyperbolic tangent function (tanh)'''
        activation = (((math.e**self.input)-(math.e**(-1*self.input)))/((math.e**self.input)+(math.e**(-1*self.input))))
        self.output = activation
        return activation

    def setType(self, type):
        self.type = type

    def __repr__(self):
        return self.type

#Testing samples:
x = Network([3,2,1])
x.feedForward([2,3,1])
x.backpropagate([6])
##y = Network([2, 1])
##x.feedForward([3,2,0.6])

##z = Network([7,3,6])
