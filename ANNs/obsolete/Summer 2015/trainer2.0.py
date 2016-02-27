#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      berejnoibejaranoc
#
# Created:     21/05/2015
# Copyright:   (c) berejnoibejaranoc 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from NeuralNetwork2 import Network


class Trainer(object):

    def __init__(self, topology,training_set, array_outputs, learning_rate = 0.01):
        self.topology = topology
        self.Lr = learning_rate
        self.network = None
        self.training_set = training_set
        self.training_outputs = array_outputs
        
        self._createSampleNet()
        
    def _createSampleNet(self):
        self.network = Network(self.topology, self.Lr)
        
    def train(self):
        
        for rep in range(1000):
            for pattern,realOut in zip(self.training_set, self.training_outputs):
                predOut = self.network.feedForward(pattern)
                if realOut != predOut:
                    self.network.backpropagate(realOut)
                    
        return self.network

def test():
#    x = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
#    y = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#    f = Trainer([4,2,1],x,y)

    x = [[1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0]]
    y = [[1], [0], [0], [0], [0], [0], [0]]
    f = Trainer([3, 2, 1], x, y, 0.1)

    net = f.train()
    
    return net
            
            
            
g = test()
