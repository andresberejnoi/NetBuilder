# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:12:11 2017

@author: andresberejnoi
"""

from distutils.core import setup
desc = """Neural Network Builder
The neural network class is in NeuralNet.py It allows to easily create fully connected feedforward networks of any size allowed by available memory. 
It uses numpy arrays as the primary data structure for the weight matrices. 
With this class you can create deep neural networks very quickly (see some of the example files to see how to use it).
"""
setup(name='NetBuilder',
      version='0.1.0',
      packages=['ANNs'],
      author='Andres Berejnoi',
      author_email='andresberejnoi@gmail.com',
      url='https://github.com/andresberejnoi/machineLearning',
      license='MIT',
      long_description=desc,
      keywords=['Neural Network', 'machine learning', 'AI', 'artificial intelligence']
      )