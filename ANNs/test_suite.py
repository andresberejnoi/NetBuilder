#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Carlos Andres Berejnoi Bejarano
#
# Created:     28/12/2015
# Copyright:   (c) PcNueva CoreI7 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#-----------OUTDATED-----------------------------------


from NeuralNet import network
import numpy as np


def main():
    net = network([2,3,2,2])
    net.setup()
    inVector = [0,1]                # input for the network. it is a python list but it should be converted to a numpy array inside the network class.
    target = np.array([0.8,0])
    output = compute_output(net,inVector)
    print_target(target)


    net.target = target

    calc_change(net)



    adjust_weights(net,net._outGradient)

    output = compute_output(net,inVector)
    print_target(target)

    for i in range(100):
        out = compute_output(net,inVector)
        calc_change(net)
        print("out gradient is: ", net._outGradient)
        adjust_weights(net,net._outGradient)

    out = compute_output(net,inVector)
    print_target(target)


    print ("End of test file")



def test_error():
    """"""
def test_setup(net):
    """"""
    net.setup()

def calc_change(net):
    """
    A shortcut function to compute the gradient of the output layer in the network
    """
    net.out_delta()
    net.outGradient()
    return net._outGradient

def test_feed(net):
    """"""
    inVector = np.random.randn(net.topology[0])


def adjust_weights(net,gradients):
    """
    Adjusts the weights for the final layer connecting to the output layer.
    gradients: a numpy array of the gradients for the output layer only. This is just
                a test function and should be improved later
    """
    print("Old weights:\n",net.weights[-1])
    print("Gradient:\n",gradients)
    net.weights[-1] = net.weights[-1] + gradients

    print("New weights:\n", net.weights[-1])

def compute_output(net, inVector):
    """A shortcut function that gives an input to the neural network to compute the output and
    then it prints it to the screen.
    net: network object
    inVector: the vector that will be fed into the network"""

    net.feedforward(inVector)

    print("Output: ", net.out)

    return net.out


def print_mat_size():
    """should print the matrix size"""

def print_target(target):
    """
    A shortcut function to print the target values for the network
    """

    print("Target: ",target)

def main2():
    net = network([2,3,2])
    net.setup()

    inVector = np.array([0.1,0.7])
    target = np.array([1,0])

    out = compute_output(net, inVector)
    net.target = target
    print_target(target)

    gradient = calc_change(net)
    print("out gradient is: ", gradient)




##if __name__ == '__main__':
##    main()

#main2()


net = network([2,3,2])
net.setup()

inVector = np.array([0.1,0.7])
target = np.array([1,0])

out = compute_output(net, inVector)
net.target = target
print_target(target)

gradient = calc_change(net)
print("out gradient is:\n", gradient)

adjust_weights(net,gradient)
