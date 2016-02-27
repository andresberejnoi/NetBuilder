import math
import matrixVector as matrix
from vector import Vector
#import numpy as np


#The following functions were taken from an online source and modified slightly to be able to use the math module instead of numpy

def sigmoid(x):
    return 1.0/(1.0 + math.e**(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return math.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2           #It is assuming x is already the tanh of some value.

class Network(object):
    def __init__(self, topology, learning_rate = 0.01, activHidden = 'tanh', activOut = 'sigmoid'):

        self.hiddenActi = tanh
        self.hidden_prime = tanh_prime
        self.outActi = sigmoid
        self.out_prime = sigmoid_prime
        
        self.topology = topology
        self.Lr = learning_rate
        self.weights = []
        self.prevInputs = []
        for i in range(1, len(topology)):
            self.weights.append(matrix.randFloatMatrix(topology[i], topology[i-1], -1, 1))
        
    def feedForward(self, inputs):                                  #It is assuming that inputs is a Matrix object already
        assert(len(inputs)==self.topology[0])
 #       print(type(inputs) == matrix.Matrix)                              #This line is for debuging. It might go away later
 #       print(type(inputs))
        
        #-----------------
        #Generating the input for the next layer
#        i = 0                                       # i represents the index of the current matrix in self.weights
#        for mat in self.weights:
#            new_matrix = mat*inputs
#            new_outs = []
#            column = new_matrix.get_colvector(0)
#            for comp in column:
#                if i != len(self.weights)-1:
#                    new_outs.append(Vector([self.hiddenActi(comp)]))
#                else:
#                    new_outs.append(Vector([self.outActi(comp)]))
#            inputs = matrix.Matrix(new_outs)
#            i += 1
#        
#        return inputs.get_colvector(0)                       #This is now a Vector object
        
#        i = 0
#        for matx in self.weights:
#            if i < len(self.weights)-1:
#                
##                new_inputVector.append(self.hiddenActi(vector.dot(inputs)))
#            else:
#                new_inputVector.append(self.outActi(vector.dot(inputs)))
#            inputs = Vector(new_inputVector)
#            i += 1
#        return inputs

        i = 0
        self.prevInputs = []
 #       self.inputs.append(inputs)                              #This is the pattern presented. It is repeated information. 
        for matx in self.weights:
            new_inputVector = []
            imp = []                        #Will contain the initial inputs before passing them through the activation function
            for vector in matx:
                inpt = vector.dot(inputs)
#                imp.append(inpt)
                if i < len(self.weights)-1:
                    new_inputVector.append(self.hiddenActi(inpt))
                else:
                    new_inputVector.append(self.outActi(inpt))
            inputs = Vector(new_inputVector)
            self.prevInputs.append(inputs)
            i += 1
        return inputs
    
    def backpropagate(self, training_set, epochs = 1):
  #      assert(type(targetOut)==Vector)
        E = 100.0                                       #Mean Error Square
        error_threshold = 0.9
        for epoch in range(epochs):
            for pattern,target in training_set:
                output = self.feedForward(pattern)
                deltaVector = target - output
                errorVector = Vector([deltaVector[i]*(output[i]*(1.0-output[i])) for i in range(len(output))])              #Calculating the delta error for the output layer
                print('length errorVec: '+ str(len(errorVector)))
                print('length inputVec: '+ str(len(self.prevInputs[::-1][0])))
                self.prevInputs = self.prevInputs[len(self.prevInputs)-2::-1]                   #This flips the list and at the same times leaves out the last element of the original unflipped list.
                self.prevInputs.append(pattern)
                flipped_weights = self.weights[::-1]
                for i in range(len(flipped_weights)):
                    deltaWeight = self.Lr*(errorVector*self.prevInputs[i])                      #This is a vector
                    for j in range(len(flipped_weights[i])):
                        flipped_weights[i][j] += deltaWeight                                    #Adds the delta weight vector to the current vector in the matrix. For now the delta weight is the same for every vector
                        #--------------------
                        #Now, we need to calculate the error vector
                        #for the current layer
                        errorVector = []
#                for i in range(len(flipped_weights)):
#                    new_vec = []
#                    for j in range(self.prevInputs[i]):
#                        new_vec.append()
#                        
                    
                    
                    
                    
                    
            
    
    
    
#Testing
#Test vectors:
inputs = matrix.Matrix([[2], [1], [0]])
vecInput = Vector([2, 0.9, -1.83])
a = Network([3, 6, 1])
a.feedForward(vecInput)


#Training stuff
x = Vector([1, 1])
y = Vector([0, 0])
z = Vector([1, 0])
w = Vector([0, 1])

b = Network([2, 8, 6, 1])
tset = [(Vector([1, 0]), Vector([1])), (Vector([0, 1]), Vector([1])), (Vector([0, 0]), Vector([0])), (Vector([1, 1]), Vector([0]))]
print(x, b.feedForward(x))
print(y, b.feedForward(y))
print(w, b.feedForward(w))
print(z, b.feedForward(z))

b.backpropagate(tset)
