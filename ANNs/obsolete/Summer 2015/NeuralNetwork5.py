import math
import matrixVector as matrix
from vector import Vector
import random
#import numpy as np
#
def randFloatMatrix(numRows, numCol, bottom_range = -1.0, top_range= 1.0):
        '''Creates a linear array of randomly generated numbers in the range of -1.0 to 1.0.
        The matrix is implemented as a 2-dimensional Python array.'''
        matrix = []

        for i in range(numRows):
            random.uniform(-10, 10)
            matrix.append([])
            for j in range(numCol):
                matrix[i].append(random.uniform(bottom_range,top_range))         #Assigns random weights to each connection

        return matrix

#The following functions were taken from an online source and modified slightly to be able to use the math module instead of numpy

def sigmoid(x):
    return 1.0/(1.0 + math.e**(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return math.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2           #It is assuming x is already the tanh of some value.

class NetworkError(Exception):
    def __init__(self, message = 'Somethig went wrong'):
        self.message = message
    def __str__(self):
        return self.message

    __repr__ = __str__

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
        self.biases = []
        for i in range(1, len(topology)):
            self.weights.append(randFloatMatrix(topology[i], topology[i-1], -1, 1))
            self.biases.append(randFloatMatrix(topology[i], 1, -1, 1))

    def feedForward(self, inputs):                                  #It is assuming that inputs is a Matrix object already
        assert(len(inputs)==self.topology[0])

#        assert(type(inputs) == Matrix)
#        assert(len(inputs[0]) == 1)
        output = []

        iter = 0                                                            # index to keep track of the iteration number
        for mat in self.weights:
            output = []
            for i in range(len(mat)):
                out = 0.0
                for j in range(len(mat[i])):
                    print(j, iter)
#                    print(inputs[j][0], self.biases[iter][j][0])
                    try:
                        out += mat[i][j]*inputs[j][0] + self.biases[iter][j][0]
                    except:
                        print(type(out))
                        print(mat[i][j])
                        print(self.biases[iter][j][0])
                       # raise NetworkError('i: ' + str(type(i))+ ', j: ' + str(type(j)) + ', iter: ' + str(type(iter)))

                if iter == len(self.weights)-1:
                    output.append(self.outActi(out))
                else:
                    output.append(self.hiddenActi(out))

            self.prevInputs.append(output)
            inputs = output
            iter += 1
        return inputs


#        for Mat, BiasVec in zip(self.weights, self.biases):                  # W is a matrix between the current layer and the previous one
#            if i == len(self.weights) -1:
#                output = [self.outActi((W*inputs)+ b) for W,b in zip(Mat, BiasVec) ]
#            else:
#                output = [self.hiddenActi]





#        i = 0
#        self.prevInputs = []
# #       self.inputs.append(inputs)                              #This is the pattern presented. It is repeated information.
#        for matx in self.weights:
#            new_inputVector = []
#            imp = []                        #Will contain the initial inputs before passing them through the activation function
#            for vector in matx:
#                inpt = vector.dot(inputs)
##                imp.append(inpt)
#                if i < len(self.weights)-1:
#                    new_inputVector.append(self.hiddenActi(inpt))
#                else:
#                    new_inputVector.append(self.outActi(inpt))
#            inputs = Vector(new_inputVector)
#            self.prevInputs.append(inputs)
#            i += 1
#        return inputs
#
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
inputs = randFloatMatrix(3, 1, 1, 5)
#vecInput = Vector([2, 0.9, -1.83])
a = Network([3, 6, 1])
a.feedForward(inputs)


#Training stuff
##x = Vector([1, 1])
##y = Vector([0, 0])
##z = Vector([1, 0])
##w = Vector([0, 1])

##b = Network([2, 8, 6, 1])
##tset = [(Vector([1, 0]), Vector([1])), (Vector([0, 1]), Vector([1])), (Vector([0, 0]), Vector([0])), (Vector([1, 1]), Vector([0]))]
##print(x, b.feedForward(x))
##print(y, b.feedForward(y))
##print(w, b.feedForward(w))
##print(z, b.feedForward(z))

#b.backpropagate(tset)
