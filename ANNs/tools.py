# -*- coding: utf-8 -*-
"""
Spyder Editor

Useful functions
"""

"""
#@author: andresberejnoi

Purpose: Some useful functions. I still need to add comments to this. 
        Some of them will probably be moved to the main NeuralNet.py file later
"""
import numpy as np

def clean():
    pass


def read_weights(weights_file, is_transpose = True, add_bias= True):
        """
        Loads weights from a text file.
        It needs to be organized better. 
        """
        from itertools import islice       
        handler = open(weights_file, 'rb')                  # opening file in binary read mode
        '''
        temp_topo = handler.readline().rstrip('\n').split(',')
        topology = []
        for item in temp_topo:
            try:
                topology.append(int(item))
            except ValueError:
                break               # it is breaking here with the assumption that all the following items are the result of the trailing commas from the csv file
        '''
        info_line = np.genfromtxt(islice(handler,0,1), delimiter=',')           # the info line is read and stored as an ndarray, but it has trailing 'nan' values from the trailing commas
        topology = info_line[np.logical_not(np.isnan(info_line) ) ]             # removes the trailing commas
        topology = topology.astype(int)         # converting the array values into integers 
        
        #print (topology)
        
        #
        #Now we read the weights based on the parameter 'is_transpose'
        #
        weights = []
        if is_transpose is False:
            # if is_transpose is false, then we read each matrix as rows=nodes in current layer,
            # and columns = nodes in following layer. If is_transpose is true, then the opposite is done
            
            for i in range(len(topology)-1):
                #weights.append(np.genfromtxt(islice(handler, 0,topology[i]), delimiter=',', usecols = range(0,topology[i+1]) ))
                #print(topology[i])
                read_until_row = int(topology[i])
                M = np.genfromtxt(islice(handler, read_until_row), delimiter=',', usecols = range(int(topology[i+1]) ) )
                M = np.atleast_2d(M)
                weights.append(M)
                #weights.append(np.genfromtxt(islice(handler, read_until_row), delimiter=',', usecols = range(int(topology[i+1]) ) ) )
            
                #print (weights)
            
        else:
            # we go here when is_transpose is true
            for i in range(len(topology) - 1):
                #print("="*80)
                #print("i: ",i)
                read_until_row = int(topology[i+1])                 # Determines until what row the file should be sliced
                #print("Rows to read: ",read_until_row)
                M = np.genfromtxt(islice(handler, read_until_row), delimiter=',', usecols = range(int(topology[i])))
                M = np.atleast_2d(M)            # this ensures that the resulting vector will always be a 2D matrix (if it is a single row, then the shape will be something like (1,20) for example.)
                weights.append(M)
                #weights.append( np.atleast_2d(np.genfromtxt(islice(handler, read_until_row), delimiter=',', usecols = range(int(topology[i])) )  )
                
                # The network loads matrices as (nodes in current layer X nodes in next layer),
                # therefore, after reading the file, we need to store matrices as their transpose forms:
                
            weights = [Mat.transpose() for Mat in weights]
                
                
        '''
        #Testing Below:
        for Mat in weights:
            print("Mat shape: ", Mat.shape)
        '''
        
        if add_bias is True:
            '''Add bias'''
            for k in range(len(weights)):
                #print("="*80)
                #print('Row: ', k)
                #print("Mat shape: ", weights[k].shape)
                #print("\n Mat:")
                #print(weights[k])
            
                bias_row = np.random.normal(scale=0.1, size=(1,topology[k+1]))          # creates a row of random values and the correct size
                #print("\nBias row: ")                
                #print(bias_row)
                
                weights[k] = np.vstack([weights[k], bias_row])                             # appends a new row to the weights file
                
        handler.close()                 #closing file
        return topology, weights

def save_outputs(output_file,Patterns, net):
    """
    It takes patterns and performs feedforward passes on it, and the output is saved to a file.
    
    output_file: a string; this is how the output file will be named. It should be a .csv file to be
                compatible with the network class at the moment.
    patterns: a list of patterns, or a str of the filename where the patterns should be read from
    
    """
    handler = open(output_file, 'wb')
    if type(Patterns) == str:
        # if the value provided for patterns is a str, the function will attempt to open it as a 
        pattern_file = open(Patterns, 'r')             # if the file does not exit, the exception FileNotFoundError will be raised
        
        for line in pattern_file:
            pattern = [float(num) for num in line.rstrip('\n').split(',')]
            out = np.atleast_2d(net.feedforward(pattern))          # computes the output
            np.savetxt(handler, out, delimiter=',')     # output is saved to file
        pattern_file.close()
    else:
        
        for pattern in Patterns:
            out = np.atleast_2d(net.feedforward(pattern))
            print(out.shape)
            np.savetxt(handler, out, delimiter=',')
    
    handler.close()
    


'''

"""
The following lines of code were taken from: 
http://stackoverflow.com/questions/32027015/how-to-read-only-specific-rows-from-a-text-file
"""

import numpy as np
import itertools
with open('test.dat') as f_in:
    x = np.genfromtxt(itertools.islice(f_in, 1, 12, None), dtype=float)
    print (x[0,:])
'''
