# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:20:11 2017

@author: Andres Berejnoi
"""
import yaml
import os

def load_model(weights_file, is_transpose = True, add_bias= True):
        """
        Loads weights from a text file.
        It needs to be organized better. 
        """
        from itertools import islice       
        handler = open(weights_file, 'rb')                  # opening file in binary read mode

        info_line = np.genfromtxt(islice(handler,0,1), delimiter=',')           # the info line is read and stored as an ndarray, but it has trailing 'nan' values from the trailing commas
        topology = info_line[np.logical_not(np.isnan(info_line) ) ]             # removes the trailing commas
        topology = topology.astype(int)         # converting the array values into integers 
        
        #Now we read the weights based on the parameter 'is_transpose'
        weights = []
        if is_transpose is False:
            # if is_transpose is false, then we read each matrix as rows=nodes in current layer,
            # and columns = nodes in following layer. If is_transpose is true, then the opposite is done
            for i in range(len(topology)-1):
                read_until_row = int(topology[i])
                M = np.genfromtxt(islice(handler, read_until_row), delimiter=',', usecols = range(int(topology[i+1]) ) )
                M = np.atleast_2d(M)
                weights.append(M)
        else:
            # we go here when is_transpose is true
            for i in range(len(topology) - 1):
                read_until_row = int(topology[i+1])                 # Determines until what row the file should be sliced
                M = np.genfromtxt(islice(handler, read_until_row), delimiter=',', usecols = range(int(topology[i])))
                M = np.atleast_2d(M)            # this ensures that the resulting vector will always be a 2D matrix (if it is a single row, then the shape will be something like (1,20) for example.)
                weights.append(M)                
                # The network loads matrices as (nodes in current layer X nodes in next layer),
                # therefore, after reading the file, we need to store matrices as their transpose forms:
                
            weights = [Mat.transpose() for Mat in weights]
        
        if add_bias is True:
            '''Add bias'''
            for k in range(len(weights)):
                bias_row = np.random.normal(scale=0.1, size=(1,topology[k+1]))          # creates a row of random values and the correct size
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
    
def save_model(dirname,model,csv_mode=False):
    """
    dirname: str; A folder named dirname will be created under the working directory and will contain network files.
    model: Network; the network to save to a file.
    """

    #pass
    
    #Save weight
    try:
        if csv_mode:
            #save weights in .csv format
            file_to_save = net.name + '_weights.csv'
            with open(file_to_save,'w') as f:
                for mat in net.weights:
                    #np.savetxt(f,mat.shape,delimiter=',')
                    np.savetxt(f,mat,delimiter=',')
                    
        else:
            #generate array names to save:
            names = [str(i) for i in range(net.size)]
            file_to_save = net.name + '_weights.npz'
            mapped_names = {key:mat for key,mat in zip(names,net.weights)}
            np.savez(file_to_save, **mapped_names)
        
        print("""Weights saved successfully in file {0}""".format(file_to_save))
    except:
        print("Something went wrong when saving weights")
        raise
    
    #Extract other network parameters:
    #hidden activation function
    #output activation function
    #name 
    #topology
    #learning rate
    #momentum
    #size
    parameters = net._get_model()
    parameters['WeightsFile'] = file_to_save    #adding the filename to the dictionary
    
    with open("""network_{0}.cfg""".format(net.name), 'w') as f:
        yaml.dump(data=parameters,stream=f)
    
    
    
        
        
        
        
        
        
        
        
        
        
        