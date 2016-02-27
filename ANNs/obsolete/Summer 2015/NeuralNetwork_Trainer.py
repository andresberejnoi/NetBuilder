from NeuralNetwork2 import Network



class Trainer:
    '''Trains a feedforward neural network using backpropagation.'''
    
    def __init__(self, topology, training_set, array_outputs, learning_rate=0.01, bias = 1):
        ''''''
        self.topology = topology
        self.training_set = training_set
        self.training_outputs = array_outputs
        self.Lr =learning_rate
        self.bias = bias
        self.Network = None
        self._createSampleNet()
        
    def _createSampleNet(self):
        self.Network = Network(self.topology, self.Lr)
    
    def train(self):
        ''''''
        for rep in range(10000):
            for pattern,output in zip(self.training_set, self.training_outputs):
                predicted_out = self.Network.feedForward(pattern)
                if predicted_out != output:
                    self.Network.backpropagate(output)
                
        return self.Network
#  x = Trainer([2,1],[[0.35,0.9]],[
def test():
#    x = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
#    y = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#    f = Trainer([4,2,1],x,y)

    x = [[0.3, 7, 1], [0, 0, 0], [10, 10, 10], [200, 150, 46], [1, 2, 4], [3, 2, 4]]
    y = [[2], [0], [1], [100], [2], [5]]
    f = Trainer([3, 2, 1], x, y)

    net = f.train()
    
    return net
            
            


net = test()
