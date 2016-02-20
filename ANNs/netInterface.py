#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Carlos Andres Berejnoi Bejarano
#
# Created:     05/01/2016
# Copyright:   (c) PcNueva CoreI7 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from NeuralNet import *
import numpy as np


def clear_input(user_input):
    """
    Cleans user input and returns a useable data structure for the network.
    user_input: a string obtained from the input method with numbers separeted by commas.

    returns: a python list of the numbers separated by commas in the raw user input."""

    clean = user_input.split(",")
    clean = [float(layer) for layer in clean]
    return clean

def process_topology(user_input):
    """
    Cleans user input when deciding the structure of the neural network.
    user_input: a string obtained from the input method that states the structure of the network

    returns: a python list where each element is the number of neurons in that layer of the neural network,
                starting with the input layer and finishing with the output layer."""

    topology = user_input.split(",")
    topology = [int(layer) for layer in topology]
    return topology

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

    returns: a numpy array of the user input"""

    target = input("""Enter expected values for each neuron in output layer, separated by a comma: """)
    target = clear_input(target)

    return np.array(target)

def main():
    """"""
    raw_topology = input("What is the topology of the network?\nEnter the number of input neurons, hidden one, hidden two, ..., output neurons. i.e: 2,3,1: ")
    topology = clear_input(raw_topology)

    print("Structure of network is: ", topology)

    net = network(topology)                 # network is initialized
    net.setup()                             # initial parameters are initialized

    inVector = get_input()

    # Getting initial output from the network by feeding it.
    actual_out = net.feedforward(inVector)

    print("Initial output of network:\n",actual_out)

    target = get_target()                   # gets the target values for the network





if __name__ == '__main__':
    main()

