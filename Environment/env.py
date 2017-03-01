# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:54:11 2016

@author: andresberejnoi
"""
import sys
import numpy as np

sys.path.append("~/Dropbox/Github/AI/ANNs/")







class genome(object):
    '''Each genome is an organism'''
    def __init__(self):
        self.fitness = 0
        self.life = 100
        

class world (object):
    #def __init__(self, agents):
      #  self.agents = np.array(agents)

    def __init__(self, height=1000,width=1000):
        
        self.map = np.zeros((height,width))
        

    def _computeFitness(self):
        """
        
        """
        
        
    
    def setupGA(self, numAgents=100, mRate=0.01, crossover=0.2, surviveRate=0.5 ):
        """
        numAgents: int; number of agents in the simulation 
        mRate: float [0.0,1.0]; the mutation rate. It is 1% by default    
        crossover: float; percentage of the population that will breed and produce offsprings (I may remove this later)
        surviveRate: float; percentage of organisms from the current generation that will survive to the next. The remaining percentage will come from new agents (random, crossover, mutation,etc)
        """
        self.numFeatures = 4
        self.agents = np.arange(1,numAgents)    # this will contain the agents, but it may end up being a dictionary for better access to attributes
        self.feature_matrix = np.zeros((self.numFeatures,numAgents))