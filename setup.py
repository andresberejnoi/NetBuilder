# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:12:11 2017

@author: andresberejnoi
"""

from setuptools import setup, find_packages
import os
from netbuilder import __version__

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

long_desc = """Neural Network Builder
The neural network class is in NeuralNet.py. It allows to easily create fully connected feedforward networks of any size allowed by available memory.
It uses numpy arrays as the primary data structure for the weight matrices.
With this class you can create deep neural networks very quickly (see some of the example files to see how to use it).
"""
desc = """Allows to create and train fully connected feedforward deep neural networks in a simple way."""
#with open('VERSION', 'r') as ver:
#    version = ver.read().rstrip()
setup(name='netbuilder',
      version=__version__,
      #packages=['NetBuilder'],
      packages=find_packages(),
      #package_data={'':paths},
      #package_data={'NetBuilder':find_packages('.')},
      include_package_data=True,
      author='Andres Berejnoi',
      author_email='andresberejnoi@gmail.com',
      url='https://github.com/andresberejnoi/NetBuilder',
      download_url = 'https://github.com/andresberejnoi/NetBuilder/archive/v0.2.3.tar.gz',
      license='MIT',
      description=desc,
      long_description=long_desc,
      keywords=['Neural Network', 'machine learning', 'AI', 'artificial intelligence', 'MLP'],
      install_requires=['numpy>=1.12','PyYAML>=3.12'],#,'sphinx>=1.5.5'],
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Development Status :: 3 - Alpha',
                   'Operating System :: Unix',
                   'Intended Audience :: Education',
                   'Intended Audience :: Developers',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
