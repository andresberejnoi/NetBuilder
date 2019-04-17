#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 02:25:03 2017

@author: andres
"""

import .netbuilder as nb
import numpy as np
import sys

def create_dummy_data():
    #Create training sets
    T,F = 1.,-1.
    input_set = np.array([[F,F],
                          [F,T],
                          [T,F],
                          [T,T]])

    target_set = np.array([[F],
                           [F],
                           [F],
                           [T]])


def test_sin_approximation():
    x = np.linspace(-5,5,200)
    y = np.sin(x)
    y = np.reshape(y,y.shape[0],1)

    #---Create network
    topology
    net = nb.Network()
    net.init()


if __name__=='__main__':
    top = [2,5,1]
    net = nb.Network()
    net.init(top)

    #get data:
    try:
        data_file = sys.argv[1]

    except IndexError:
        data_file =
    #Create training sets
    T,F = 1.,-1.
    input_set = np.array([[F,F],
                          [F,T],
                          [T,F],
                          [T,T]])

    target_set = np.array([[F],
                           [F],
                           [F],
                           [T]])

    net.train(input_set=input_set,
              target_set=target_set,
              batch_size=0,
              epochs=1000,
              print_rate=100,
              plot=True)
