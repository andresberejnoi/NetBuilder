#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Carlos Andres Berejnoi Bejarano
#
# Created:     10/01/2016
# Copyright:   (c) PcNueva CoreI7 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from NeuralNet import *
from NetworkTrainer import *

def get_topology():
    raw_topology = input("What is the topology of the network?\nEnter the number of input neurons, hidden one, hidden two, ..., output neurons. i.e: 2,3,1: ")
    topology = clear_input(raw_topology)
    print("Structure of network is: ", topology)
    return topology

def clear_input(user_input):
    """
    Cleans user input and returns a useable data structure for the network.
    user_input: a string obtained from the input method with numbers separeted by commas.

    returns: a python list of the numbers separated by commas in the raw user input."""

    clean = user_input.split(",")
    clean = [float(layer) for layer in clean]
    return clean

def get_input():
    """
    Gets user input to feed the neural network. It assumes that the input provided
    corresponds to only one training example.

    returns: a numpy array with the input to the network. """

    inVector = input("""Enter inputs to each neuron in input layer, separated by a comma: """)

    vector = clear_input(inVector)                  # a python list of the user input
    return np.array(vector)                         # converts the input list into a numpy array and returns it

def get_target():
    """
    Asks user to provide the expected values for the output to the network.
    target values are entered one per output neuron separated by a comma. A single
    training example is assumed.

    returns: a numpy array of the user input
    """

    target = input("""Enter expected values for each neuron in output layer, separated by a comma: """)
    target = clear_input(target)

    return np.array(target)


def main1():
    #topology = get_topology()
    topology = [2,3,1]
    net = network(topology)
    net.setup()

    ##trainingSize = int(raw_input("Number of training samples: "))

    #print(net.weights)                                          # DEBUGGING FOR ONLY

    # Inputs:
    x1 = np.array([0,0])
    x2 = np.array([0,1])
    x3 = np.array([1,0])
    x4 = np.array([1,1])


    trainingInputs = [x1,x2,x3,x4]
    trainingTargets = [np.array([0]),
                        np.array([1]),
                        np.array([1]),
                        np.array([0])]

    # Initializing a network trainer:
    netTrainer = trainer()

    ### create a for loop to get as many training samples as needed.
    ##print("Enter the training set inputs and targets:")
    ##for i in range(trainingSize):
    ##    trainingInputs.append(get_input())
    ##    trainingTargets.append(get_target())

    #----------------------------------------------------------------
    # Print useful information
#    print("Training Inputs:\n",trainingInputs)
#    print("Training Targets:\n",trainingTargets)
    #----------------------------------------------------------------


    netTrainer.set_trainingInputs(trainingInputs)
    netTrainer.set_trainingTargets(trainingTargets)


    net = netTrainer.train_EXP(net)

    return net

def main2():
    topology = get_topology()
    net = network(topology)
    net.setup()

    trainingSize = int(input("Number of training samples: "))

    trainingInputs = []
    trainingTargets = []

    # Initializing a network trainer:
    netTrainer = trainer()

    # create a for loop to get as many training samples as needed.
    print("Enter the training set inputs and targets:")
    for i in range(trainingSize):
        trainingInputs.append(get_input())
        trainingTargets.append(get_target())

    #----------------------------------------------------------------
    # Print useful information
    print("Training Inputs:\n",trainingInputs)
    print("Training Targets:\n",trainingTargets)
    #----------------------------------------------------------------


    netTrainer.set_trainingInputs(trainingInputs)
    netTrainer.set_trainingTargets(trainingTargets)


    net = netTrainer.train_EXP(net)

    return net




net = main1()

