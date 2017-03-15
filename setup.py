# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:12:11 2017

@author: andresberejnoi
"""

from setuptools import setup
long_desc = """Neural Network Builder
The neural network class is in NeuralNet.py. It allows to easily create fully connected feedforward networks of any size allowed by available memory. 
It uses numpy arrays as the primary data structure for the weight matrices. 
With this class you can create deep neural networks very quickly (see some of the example files to see how to use it).
"""
desc = """Allows to create and train fully connected feedforward deep neural networks in a simple way"""
with open('VERSION', 'r') as ver:
    version = ver.read().rstrip()
setup(name='netbuilder',
      version=version,
      packages=['NetBuilder'],
      author='Andres Berejnoi',
      author_email='andresberejnoi@gmail.com',
      url='https://github.com/andresberejnoi/NetBuilder',
      license='MIT',
      description=desc,
      long_description=long_desc,
      keywords=['Neural Network', 'machine learning', 'AI', 'artificial intelligence'],
      install_requires=['numpy>=1']
      )
