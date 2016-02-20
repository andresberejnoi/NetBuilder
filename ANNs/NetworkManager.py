#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      berejnoibejaranoc
#
# Created:     15/01/2016
# Copyright:   (c) berejnoibejaranoc 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------


class NetworkManager():
    """
    A collection of methods that allow the user to store the weights of a trained network into a text file, or
    retrieve values from a text file, as long as the format is the correct one.
    This manager will create a network based on the information that it reads from the file provided.
    Given that numpy already provides functions for reading and writing to an array, this class will offer mostly
    wrappers around numpy's functions, to make it more user friendly.
    """

    def __init__(self):
        """Initializer. It might not be necessary"""



    def _format_weights(self, net):
        """To be included later...
        This function might not be necessary because numpy already includes functions to save and load from a file."""

        str_weights = []                                        # it will contain the string versions of the network weights
        for W in net.weights:
            pass




    def save_network(self,net,filename):
        """
        Writes the weights of network net into filename using numpy's saving capabilities.
        net: A neural network object
        filename: python string; the filename where the information of the network should be stored

        returns: None
        """
        handler = open(filename,"w")

