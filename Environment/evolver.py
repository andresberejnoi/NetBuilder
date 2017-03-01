# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:52:44 2017

@author: andresberejnoi
"""

class networkGA(object):
    '''Evolves a fully connected neural network weights (for now)'''
    def __init__(self, numPop=100, mRate=0.01, survive=2, crossover = 0.2):
        self.numPop = numPop
        self.mrate = mRate
        self.survivors = survive
        self.crossover = crossover
        self.agents = {i:{} for i in range(numPop)}
        
        
    def get_fitness(self, ID):
        return self.agents[ID][0]
        
    def calculate_fitnesses(self):
        for agent in self.agents:
            pass

#definition of the gnome
class genome(object):
    "Needs to describe"
    def __init__(self):
        pass
                    
        