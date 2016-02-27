from neural_networks import Perceptron
import random

class perceptronTrainer(object):
    '''Trains the perceptron until it reaches a good performance'''
    
    def __init__(self, training_set, output, threshold):
        ''''''
        self.inputs = training_set                  #The set of inputs 
        self.desiredOutputs = output
        self.weights = []
        self.threshold = float(threshold)
        self.runs = 0           #The number of tests or trainning sessions performed
        self.successes = None     #Will contain the last successful perceptron
       
       
        self._generateWeights()             #generates random weights
        
    def _generateWeights(self):
        '''Randomly generates weights to use in the perceptron.
        The weights are randomly chosen in the range of -1 to 1'''
        for i in range(len(self.inputs[0])):
#            self.weights.append(round(random.uniform(-1, 1), 1))        #Gets a random weight between -1 and 1 with only 1 decimal
            self.weights.append(random.randint(-10, 10))            #The random weights are integers from -10 to 10.
   
#    def _updateWeights(self):
#        '''Updates the weights after running a test.'''
#        
#    def _getError(self, calcOutput):
#        '''Calculates the error in the outputs. It is necessary for the weight update rule. Based on
#        the book "Artificial Intelligence: A Modern Approach".'''
#        error = self.desiredOutputs[self.runs-1] - calcOutput
#        return error
#        
#    def weightUpdateRule(self):
#        '''Updates the weights.'''
#        
        
    def _compareOutputs(self, percep_output, config):
        '''Checks if the acutal output from the perceptron is the 
        same as the expected correct output'''
        if percep_output == self.desiredOutputs[self.runs-1]:
            self.successes = Perceptron(config[0], config[1])           #Creates a Perceptron with the successful combination of weights and the threshold stored in config.
            return True
        else: 
            if percep_output == 1 and self.desiredOutputs[self.runs-1]==0:
                self.threshold += 1
                for i in range(len(self.weights)):
                    if self.inputs[self.runs-1][i] == 1:
                        self.weights[i] -= 1
            else:
                self.threshold -= 1
                for i in range(len(self.weights)):
                    if self.inputs[self.runs-1][i] == 1:
                        self.weights[i] += 1
        return False

    def trainer(self): 
        '''Trains the network'''
        for rep in range(100):
            self.runs = 0
            for i in range(len(self.inputs)):
                self.runs += 1
                P = Perceptron(self.weights, self.threshold)
                result = P.activationFunction(self.inputs[i])
                self._compareOutputs(result, P.get_config())
                
            
        return self.successes
        
