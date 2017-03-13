# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:50:33 2017

@author: andresberejnoi
"""

import tensorflow as tf
import numpy as np

def sample_network():
    '''
    num_in_samp = 1
    num_in_features = 5
    
    num_out = num_in_samp
    num_out_features = 2
    
    x = np.random.rand(num_in_samp,num_in_features)
    y = np.random.rand(num_out,num_out_features)
    '''
    #defining some network parameters:
    num_in_samp = 4
    num_in_features = 2
    
    num_out = num_in_samp
    num_out_features = 1
    
    error_thresh = 0.001
    max_epochs = 10000
    
    x = np.array([[-1.,-1.],
                  [-1.,1.],
                  [1.,-1.],   
                  [1.,1.]])
                            
    y = np.array([[-1.],
                  [-1.],
                  [-1.],
                  [1.]])
    
    
    
    
    with tf.Graph().as_default():
        
        input_train = tf.placeholder(dtype=tf.float64)
        output_train = tf.placeholder(shape=(num_out,num_out_features),dtype=tf.float64)
        
        #x = tf.truncated_normal((10,10))
        
        
        #parameters that define the network:
        """
        topology = [5,5,5,5]
        
        """
        #-------------------------------------------------------
        # Creating Layers
        #Define Layer 1 parameters
        layer1 = [num_in_features,10]     #shape
        layer1_weights = tf.Variable(tf.random_normal(layer1,dtype=tf.float64))
        layer1_bias = tf.Variable(tf.random_normal([10],dtype=tf.float64))
        
        layer2 = [10,10]
        layer2_weights= tf.Variable(tf.random_normal(layer2,dtype=tf.float64))
        layer2_bias = tf.Variable(tf.random_normal([10],dtype=tf.float64))
        
        layer3 = [10,num_out_features]
        layer3_weights = tf.Variable(tf.random_normal(layer3,dtype=tf.float64))
        layer3_bias = tf.Variable(tf.random_normal([num_out_features],dtype=tf.float64))
        
        #Define the operations:
        feed1 = tf.tanh(tf.add(tf.matmul(input_train,layer1_weights),layer1_bias))
        feed2 = tf.tanh(tf.add(tf.matmul(feed1,layer2_weights),layer2_bias))
        feed3 = tf.tanh(tf.add(tf.matmul(feed2,layer3_weights),layer3_bias))
        
        print('out_train:',output_train.shape)
        print('shape 3:',feed3.shape)
        
        #computing the error:
        #error = tf.sub("expected_out, actual_out")
        #error = tf.Variable(tf.subtract(output_train,feed3))
        #mse = tf.reduce_mean(error)
        
        
        #init = tf.global_variables_initializer()
        #generating the graph
        #graph = tf.Graph()
        with tf.Session() as sess:
            feed_dict = {input_train:x,output_train:y}  
            sess.run(tf.global_variables_initializer())
            output = sess.run(feed3,feed_dict=feed_dict)    
            print('output:\n',output)
    
        
        #with tf.Session() as sess:
            #feed_dict = {output_train:y}
            
            error = tf.Variable(tf.abs(tf.subtract(output_train,output)))
            mse = tf.reduce_mean(error)
            train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)
            
            sess.run(tf.global_variables_initializer(),feed_dict=feed_dict)
            
            err, target = 1, 0.000001
            epoch, max_epochs = 0, 50000
            while err > target and epoch < max_epochs:
               epoch += 1
               err, _ = sess.run([mse, train])
               print('epoch:',epoch,'error:',err)
               if epoch%(max_epochs/10)==0:
                   print('epoch:',epoch,'error:',err)
            print('epoch:', epoch, 'mse:', err)
            
            
            output = sess.run(feed3,feed_dict=feed_dict)
            print('='*80,"\nTEST:")
            print('OUTPUT\tEXPECTED')
            for i in range(num_in_samp):
                print(output[i],y[i])
if __name__=='__main__':
    #sample_network()
    pass
    

class Network(object):
    '''A network builder that uses tensorflow as a backend.'''
    def __init__(self):
        self.name = 'Network'
        self.layers = []        # this will store the variables of weight matrices
        self.layer_biases = []  # this will store bias vectors for each layer
        self.ops = []           # this will store the operations for each layer
        self.graph = None           #this will contain a tensorflow graph
        self.actFunc = None
        self.init_vars = None
        self.input_data = None
        self.output_data = None

        
    def new(self,topology, actFunc=tf.tanh):
        '''
        Create a network with the structure specified in topology:
        topology: python list; i.e. [5,10,10,2] indicates a network with input layer of 5 neurons,
                    then 2 hidden layers with 10 neurons each, and finally an output layer with 2 neurons
                    TODO: topology can also be a filename for a configuration file that can specify more
                        things about the network.
        '''
        self.actFunc = actFunc      #this is the activation function that will be used for the operations        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_data = tf.placeholder(dtype=tf.float64)
            self.output_data = tf.placeholder(dtype=tf.float64)
            for i in range(len(topology)-1):
                print('i:',i)
                layer_shape = [topology[i],topology[i+1]]
                bias_shape = [layer_shape[1]]
                if i==0:
                    Lname = 'input_L'
                    Bname = 'input_B'
                    op_name = 'input_F'
                    op_input = self.input_data
                elif i==len(topology)-1:
                    Lname = 'output_L'
                    Bname = 'output_B'
                    op_name = 'output_F'
                    op_input = self.ops[i-1]
                else:
                    idx = i+1
                    Lname = """layer{0}""".format(idx)
                    Bname = """bias{0}""".format(idx)
                    op_name = """op_{0}""".format(idx)
                    op_input = self.ops[i-1]
                #name = """layer{0}""".format(i)
                layer = tf.Variable(tf.random_normal(shape=layer_shape,dtype=tf.float64), name=Lname)
                layer_bias = tf.Variable(tf.random_normal(shape=bias_shape,dtype=tf.float64), name=Bname)
                self.layers.append(layer)
                self.layer_biases.append(layer_bias)
                #self.layers.append(tf.Variable(tf.random_normal([topology[i],topology[i+1]],dtype=tf.float64))) 
                
                #Setup the operations
                oper = self.actFunc(tf.add(tf.matmul(op_input,layer),layer_bias),name=op_name)
                self.ops.append(oper)
                
                #create the graph object
                #self.graph = tf.Graph()
            self.init_vars = tf.global_variables_initializer()
            
    
    def feedforward(self, input_data):
        """
        Runs a TensorFlow session with with the graph in self.graph.
        input_data: ndarray; a matrix with the data that will be fed into the network        
        """
        
        feed_dict = {self.input_data:input_data}
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init_vars)
            output = sess.run(self.ops[-1],feed_dict=feed_dict)
        
        return output
    def _compute_error(self,expected_out,output):
        """
        This is a stand-alone function so that the code is more portable.        
        """
        with self.graph.as_default():
            error = tf.Variable(tf.subtract(expected_out,output))
            mse = tf.reduce_mean(error)
        
        return mse
        
    def train(self, expected_out, actual_out, error_thresh=0.0001,epochs_to_run=5000):
        #First compute the error of the network
        with tf.Session(graph=self.graph) as sess:
            feed_dict = {self.output_data:actual_out}
            #error = tf.Variable(tf.subtract(expected_out,actual_out))
            #mse = tf.reduce_mean(error)
            mse = self._compute_error(expected_out,actual_out)
            to_train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)
            
            sess.run(tf.global_variables_initializer())#,feed_dict=feed_dict)         #variables in _comput_error() will be initialized
            
            error = 1. 
            epoch = 0
            while error > error_thresh and epoch < epochs_to_run:
               epoch += 1
               error, _ = sess.run([mse, to_train])
               
               #Print information during training
               if epoch%(epochs_to_run/10)==0:
                   print('epoch:',epoch,'error:',error)
                   
            print('Finished:')
            print('epoch:', epoch, 'mse:', error)




def test1():
    num_in_samp = 1
    num_in_features = 5
    
    num_out = num_in_samp
    num_out_features = 2
    
    x = np.random.rand(num_in_samp,num_in_features)
    y = np.random.rand(num_out,num_out_features)
    
    print()
    print('='*160)
    print('NEW TEST')
    net = Network()
    topology = [num_in_features,10,10,num_out_features]
    print("Creating network of topology:",topology)
    net.new(topology)
    
    print('='*80)
    print("Starting forward propagation:")
    out = net.feedforward(x)
    print('output:',out,'\n')

    print('='*80)
    print("Starting Training:")
    net.train(y,out)
    
def test_AND():
    test_name = 'AND Test Began:'
    print('='*100,'\n','='*100,'\n',test_name)
    
    #defining some network parameters:
    num_in_samp = 4
    num_in_features = 2
    
    num_out = num_in_samp
    num_out_features = 1
    
    error_thresh = 0.001
    max_epochs = 10000
    
    input_train = np.array([[-1.,-1.],
                            [-1.,1.],
                            [1.,-1.],
                            [1.,1.]])
                            
    output_train = np.array([[-1.],
                             [-1.],
                             [-1.],
                             [1.]])
                             
    #creating the network
    topology = [num_in_features,5,5,num_out_features]
    net = Network()
    net.new(topology)
    print('='*80,'\n','Network created:',topology)
    
    #Propagating inputs feedforward and training the network
    print("TRAINING BEGINS:")
    out = net.feedforward(input_train)
    print('='*80,'\n','Feedforward finished\n','Output:\n',out)
    
    net.train(output_train,out,error_thresh=error_thresh,epochs_to_run=max_epochs)
    print("TRAINING FINISHED")
    
    #Now we test the network predictions
    print('='*80,'\n','TESTING BEGINS:\n')
    test_out = net.feedforward(input_train)
    print('Output:\n','Actual','\tExpected')
    for i in range(num_in_samp):
        print(test_out[i],output_train[i])
    

def test_XOR():
    test_name = 'XOR Test Began:'
    print('='*100,'\n','='*100,'\n',test_name)
    
    #defining some network parameters:
    num_in_samp = 4
    num_in_features = 2
    
    num_out = num_in_samp
    num_out_features = 1
    
    error_thresh = 0.001
    max_epochs = 10000
    
    input_train = np.array([[-1.,-1.],
                            [-1.,1.],
                            [1.,-1.],
                            [1.,1.]])
                            
    output_train = np.array([[-1.],
                             [1.],
                             [1.],
                             [-1.]])
                             
    #creating the network
    topology = [num_in_features,5,5,num_out_features]
    net = Network()
    net.new(topology)
    print('='*80,'\n','Network created:',topology)
    
    #Propagating inputs feedforward and training the network
    print("TRAINING BEGINS:")
    out = net.feedforward(input_train)
    print('='*80,'\n','Feedforward finished\n','Output:\n',out)
    
    net.train(output_train,out,error_thresh=error_thresh,epochs_to_run=max_epochs)
    print("TRAINING FINISHED")
    
    #Now we test the network predictions
    print('='*80,'\n','TESTING BEGINS:\n')
    test_out = net.feedforward(input_train)
    print('Output:\n','Actual','\tExpected')
    for i in range(num_in_samp):
        print(test_out[i],output_train[i])
    
    
    
    
if __name__=='__main__':
    # Run test functions
    #test1()
    #test_AND()
    #test_XOR()
    sample_network()