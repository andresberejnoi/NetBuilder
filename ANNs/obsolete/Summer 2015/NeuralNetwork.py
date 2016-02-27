import math
import random
from showNetwork import diagramNetwork as graph

class NeuralNetwork(object):
    '''Provides methods for creating and manipulating an artificial neural network.
    The network should be trained using the backpropagation method.'''
    def __init__(self, topology, learning_rate = 0.01, bias = 1):
        '''Initializes all the instance variables in the network.
        topology: a list of integers; each integer refers to the number of neurons
                    in a given layer. It goes from left to rights, [input_layer, hidden_layer,...,output_layer]'''
        self.topology = topology
        self.Lr = learning_rate
        self.bias = bias
        self.layers = []                      #an array of the layers in the network. Should have the same order as in self.topology
        self.output = None

        self._generateNetwork()               #The network is created

    def __len__(self):
        return len(self.layers)

#    def __repr__(self):
#        graph(self.topology)
#        return str(self.topology)


    def _generateNetwork(self):
        '''Generates the network, creating the layers.'''
        for i in range(len(self.topology)):
            type = ''
            length_previous = 0
            if i == 0:
                type = "I"
                length_previous = self.topology[i]
            elif i == len(self.topology)-1:
                type = "O"
                length_previous = self.topology[i]+1
            else:
                type = "H"
                length_previous = self.topology[i]+1

            newLayer = Layer(self.topology[i], type, length_previous, self.Lr, self.bias)
            self.layers.append(newLayer)
    def getOutput(self, array_inputs):
        '''Computes the output to the network
        array_inputs: an array of the inputs that are going to be fed to the network.
        expected_output: the true output that the network should return'''
        result = []
        #Feedforward step
        for layer in self.layers:
            for node in layer.nodes:
                if type(node)==int:
                    continue
                node.sigmoidActivation(array_inputs)
        for node in self.layers[-1]:
            if type(node)==int:
                continue
            self.output = node.output
         #   result.append(node.output)             #FIXXX!!!!!!!!!!!

        return self.output

    def errorProp(self, expected_output):
        ''''''
    #Calculating the error signals for the output layer:
        for node in self.layers[-1]:                #The -1 index gets the final layer in the network, which is the output layer.
            if type(node)==int:
                continue
            node._errorOutputNode(expected_output)
        lastWeights = self.layers[-1].getWeights()
        lastErrors = self.layers[-1].getErrors()

    #Backpropagation of the error:
        for layer in self.layers[::-1][1:]:             #iterates over every layer except the output layer
            counter = 0
            for node in layer:
                if type(node)==int:
                    continue
                pW = [w[counter] for w in lastWeights]                  #A list of all the weights from this node to the next layer
                error = node._errorHiddenNode(lastErrors, pW)
                node.updateWeights(error)
                counter += 1
            lastWeights = layer.getWeights()
            lastErrors = layer.getErrors()






#    def _showNetwork(self):
#        '''Uses turtles to show the network graphically.'''
#        import turtles
#        for nodes in self.layers:
#            for i in range(nodes):
#



class Layer(object):
    '''Provides the methods for instantiating a layer for a neural network'''

    def __init__(self, numNeurons, type, length_previousLayer, learning_rate = 0.01,  bias = 1):
        self.numNeurons = numNeurons
#        self.weights = array_weights
        self.type = type.upper()
        self.bias = bias
        self.previous_layer = length_previousLayer          #The number of nodes in the previous layer. It could be changed to represent a link to the actual layer
        self.Lr = learning_rate
        self.errors = None
        self.nodes = []                 #The actual "layer" of neurons, represented as an array
        self._generateLayer()

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        for node in self.nodes:
            yield node
    def __repr__(self):
        return (str(self.nodes))


    def getWeights(self):
        weights = []
        for node in self.nodes:
            if type(node)==int:
                continue
            weights.append(node.weights)
        return weights

    def getErrors(self):
        '''Returns the errors for each node as an array'''
        errors = []
        for node in self.nodes:
            if type(node)==int:
                continue
            errors.append(node.errorSignal)
        return errors

    def _generateLayer(self):
        if not self.type=="I" and not self.type == "O":
            self.nodes.append(self.bias)
        for i in range(self.numNeurons):             #The +1 is to account for the bias factor, which acts as an extra input
            newNode = Neuron(self._randomWeights(), self.type, self.Lr)
            self.nodes.append(newNode)

    def _randomWeights(self):
        '''Generates random weights from 0.20 to 0.80 and applies them to every node in the layer'''
        weights = []
        for i in range(self.previous_layer):
       #     weights.append(round(random.uniform(0.2, 0.8), 2))              #adds a random weight per every node in the previous layer. The bias is also taken into account here
            weights.append(random.uniform(0.2, 0.8))
        return weights


class Neuron(object):
    '''A single neuron in a multi-layer network.'''
    def __init__(self, array_weights, type, learning_rate = 0.01):
        '''Initializes the neuron.
        array_weights: an array of the weights from the previous layer to this neuron.
        type: a string of one character. It represents the type of neuron: "I" for input, "H" for hidden, and "O" for output.
        learning_rate = a float number'''
        self.weights = array_weights
        self.output = None
        self.type = type.upper()                    #the type will always be an uppercase letter.
        self.Lr = round(float(learning_rate), 2)
        self.input = None
        self.errorSignal = None

    def __repr__(self):
        return ("out: "+ str(self.output))

    def getWeights(self):
        '''Returns the weights from this neuron to the previous layer.'''
        return self.weights

    def _net_input(self, inputs):
        '''Calculates the net input to the current neuron.'''
#        if not self.type == "I":
        self.input = sum([W*I for W,I in zip(self.weights, inputs)])
        return self.input

    def sigmoidActivation(self, inputs):
        '''Calculates the output of the neuron using a sigmoid function.
        inputs: an array of input values from the previous layer.'''
        self.output = 1/(1+ math.e**(-1*self._net_input(inputs)))
        return self.output

    def _errorOutputNode(self, true_output):
        '''Calculates the error signal of the current neuron.
        true_output: a real number, the expected output of the neuron, according to the pattern presented'''
        self.errorSignal = 2*(true_output-self.output) * self.output*(1-self.output)              #When the node is in the output layer, the error is simply the difference between the expected and the predicted output
        return self.errorSignal                                                                     #times the derivative of the activation function used. In this case, the sigmoid function

    def _errorHiddenNode(self, next_errors, next_weights):
        '''Calculates the signal error for a hidden neuron.
        next_errors: an array of errors from the next layer in the network.
        next_weights: an array of weights connecting the current node or unit to the next layer.'''
        self.errorSignal = -1*sum([signal*weight for signal,weight in zip(next_errors, next_weights)])               #The errors of each node in the next layer are multiplied by the weights connecting the current node to the next layer, and then everything is added together
        return self.errorSignal                                                                                                 #The -1 changes the sign for the whole expression

    def updateWeights(self, error):
        '''Updates the weights to the current neuron from the previous layer'''
        for i in range(len(self.weights)):
#            if self.is_outputNode():                        #If the current node is in the output layer, then a different error function calculation is needed
#                self.weights[i] += self.Lr*self._errorOutputNode()
#            else:
#                self.weights[i] += self.Lr*self._errorHiddenNode()
            if self.is_outputNode():                        #If the current node is in the output layer, then a different error function calculation is needed
                self.weights[i] += self.Lr*self._errorOutputNode()*self.input
            else:
                self.weights[i] += self.Lr*error*self.input

    def get_type(self):
        '''Returns the type of the neuron.'''
        return self.type

    def is_outputNode(self):
        '''Returns True if the node is an output node. Returns False otherwise.'''
        if self.type == "O":
            return True
        return False








