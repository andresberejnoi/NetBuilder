# -*- coding: utf-8 -*-
"""Created on Mon Mar 20 11:20:11 2017

@author: Andres Berejnoi
"""
#from . import NeuralNet
#from netbuilder import Network
#from . import keys
#from . import Network, keys
from netbuilder import keys
import numpy as np
#from netbuilder import np
import yaml
import os


def load_model(directory,is_csv=False):
    """Loads a network model that is saved in the specified directory.

    Parameters
    ----------
    directory : str
        Folder path where network save files are stored.
    is_csv : bool, not implemented
        A boolean flag to know if the model to load uses csv or numpy format for the weights.
    """

    #remember current directory and move to desired directory
    start_dir = os.getcwd()
    os.chdir(directory)

    #look for configuration file
    config_file = keys._config_file    #I will look for a better way to automate this file name or make it accessible across the package
    with open(config_file,'r') as f:
        parameters = yaml.load(f)
    #name = parameters['name']
    #topology = parameters['topology']
    #learningRate = parameters['learningRate']
    #momentum = parameters['momentum']
    #hidden_activation = parameters['hiddenActivation']
    #output_activation = parameters['outputActivation']
    #size = parameters['size']
    weights_file = parameters[keys._weights_file]

    #open network weights
    weights_dict = None
    if is_csv:
        pass
    else:
        weights_dict = np.load(weights_file)

    #Go back to starting directory
    os.chdir(start_dir)

    #Create and initialize network
    #net = Network()
    net = netbuilder.Network()
    net._init_from_file(params=parameters,weights_dict=weights_dict)

    print("Model {0} loaded correctly".format(net.name))

    return net


def save_model(net,directory='.',csv_mode=False):
    """Creates a directory and saves the network model in it.

    Parameters
    ----------
    directory : str
        Directory where network save folder will be created.
    model : Network
        Network object to save to a file.
    csv_mode : bool, not implemented
        if True then save network weights as a csv file. Otherwise, weights are saved as numpy format *.npz.

    Returns
    -------
    str
        The path to the output folder so that it can be loaded later.
    """

    folder_name_base = "{0}_Model".format(net.name)
    fold_index = _get_next_foldername_index(folder_name_base,directory)
    net_folder_name = "{0}.{1}".format(folder_name_base,fold_index)
    #pass
    initial_working_dir = os.getcwd()
    print("Working directory when calling save:",initial_working_dir)

    #move to specified directory and create output folder
    os.chdir(directory)

    try:
        os.mkdir(net_folder_name)
    except FileExistsError:
        raise
    os.chdir(net_folder_name)
    output_path = os.getcwd()

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
        #name
        #topology
        #learning rate
        #momentum
        #size
    parameters = net._get_model()
    parameters[keys._weights_file] = file_to_save    #adding the filename to the dictionary

    with open(keys._config_file, 'w') as f:
        yaml.dump(data=parameters,stream=f)


    print("Files saved successfully at location:",output_path)

    #When everthing is done, go back to original working directory
    os.chdir(initial_working_dir)

    #return the path of output folder in case it is needed later
    return output_path

def _get_next_foldername_index(name_to_check,dir_path):
    """Finds folders with name_to_check in them in dir_path and extracts which one has the hgihest index.

    Parameters
    ----------
    name_to_check : str
        The name of the network folder that we want to look repetitions for.
    dir_path : str
        The folder where we want to look for network model repetitions.

    Returns
    -------
    str
        If there are no name matches, it returns the string '1'. Otherwise, it returns str(highest index found + 1)
    """

    dir_content = os.listdir(dir_path)
    dir_name_indexes = [int(item.split('.')[-1]) for item in dir_content if os.path.isdir(item) and name_to_check in item]    #extracting the counter in the folder name and then we find the maximum

    if len(dir_name_indexes) == 0:
        return '1'
    else:
        highest_idx = max(dir_name_indexes)
        return str(highest_idx + 1)
    #find all folders that have name_to_check in them:
