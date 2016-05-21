# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 04:01:36 2016

@author: andresberejnoi
"""
from population_equationGA import Population

def main():
    """"""
    pop = Population()
    pop.evolve(iterations=15, mutation_rate=0.20)
    
    
    
if __name__ == '__main__':
    main()