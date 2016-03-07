# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:04:41 2016

@author: andresberejnoi
"""
# Import the equation class. This also imports numpy as np
from individual_equationGA import *

#==============================================================================
def __power(min_x = -20, max_x = 20, number_samples=40, coefficient =1, raise_to = 2):
    """Applies the power function to a set of numebers and returns a list of inputs and outputs"""
    
    inputs = np.linspace(min_x,max_x, number_samples)
    outputs = np.power( (coefficient*inputs), raise_to)
    
    return inputs, outputs
    
# This is a dictionary of functions to test the GA program without having a file with real data
functions = {'power':__power}

#==============================================================================
def get_fit(eq_individual):
    """Given an Equation object, it will return its fitness score."""
    return eq_individual.get_fitness()


class Population(object):
    """Methods for a population equation individuals"""
    def __init__(self, size=100, survival_rate = 0.7, filename = None):
        """
        size: an int indicating the population size (how many individuals are present
        survival_rate: a float from 0 to 1; it indicates what percentage of the population will be kept after each iteration
        """
        self.size = size
        self.survival_rate = survival_rate
        self.equations = []             #a list that will contain all the individuals in the population
        self.filename = filename
        
        self.rand = np.random.randint       # a shortcut to numpy's random.randint function. This could speed up the process
        
        #Target inputs and outputs will be loaded
        self.target_inputs = None
        self.target_outputs = None
        
        self._initialize_population()   #population is initialized with random values
        
        self._load_targets()
        self._setup()
        
    
    def _initialize_population(self):
        """Initializes individuals that will form the population"""               
        self.equations = np.array([Equation(14,self.rand(1,15)) for i in range(self.size)])
        
    def _setup(self):
        """A place to set up things at initialization"""
        
        #Vectorizing some functions:
        self.get_fitnesses = np.vectorize(get_fit)
    
        
    def _load_targets(self):
        """Sets the target values for comparison to the individuals' performance"""
        
        try:
            f_handler = open(self.filename)
        except (FileNotFoundError, TypeError):
            print("Filename for data not provided or not found.")
            message = """Please select one of the following functions:\n{0}""".format(''.join(['*\t '+fun+'\n' for fun in list(functions)]) )
            print (message)
            func_name = input("Function selected: ")
            fun = functions[func_name]
            
            use_default = input("Would you like to use the default parameters for the function? (y/n): ")
            
            if (use_default.lower() == 'y') or (use_default == 'yes') or (use_default==''):
                self.target_inputs,self.target_outputs = fun()
            else:
                min_x = float(input("Enter minumum value for x: "))
                max_x = float(input("Enter maximum value for x: "))
                num_samples = int(input("How many sample points would you like?: "))
                try:
                    coefficient = float(input("What coefficient would you like? (default is 1): "))
                except ValueError:
                    coefficient = 1
                
                # Here there should be more questions about the other parameters each function might have
                self.target_inputs,self.target_outputs = fun(min_x,max_x,num_samples,coefficient,2)
                
            
            
        
        
    def calculate_population_fitness(self):
        """Calculate the total fitness for the current population"""
        
        total_fit = 0
        for equation in self.equations:
            total_fit += equation.calculate_fitness(self.target_inputs,self.target_outputs)
        
        # take the average
        avg_fitness = total_fit/self.size
        
        return avg_fitness
        
    
    def mutation(self, mutation_prob = 0.01):
        """"""
        
        for equation in self.equations:
            equation.mutate(mutation_prob)
            
    
    def set_new_generation(self):
        """Based on the survival rate parameter, it will remove the worst performing individuals
        and replace them. This would be the work of a crossover function. However, for now, the new 
        individuals will be generated randomly"""
    
        #Calculate how many will be replaced:
        numSurvivors = int(self.size*self.survival_rate)
        left4Dead = self.size - numSurvivors
        
        # find the positions of worst individuals:
        positions = self._find_worst_performing(left4Dead)
        
        
        for idx in positions:
            self.equations[idx] = Equation(14,self.rand(1,15))
        
        
        
    


    
    def _find_worst_performing(self,num):
        """
        Finds the index of the worst performing members of the population.
        num: how many individuals should be found. It needs a lot of improvement. This is just a temporary implementaion.
        
        return: a list of the index positions of each individual
        """
        fitnesses = list(self.get_fitnesses(self.equations))
        positions = []
        for i in range(num):
            min_val = min(fitnesses)
            pos = fitnesses.index(min_val)
            positions.append(pos)
            fitnesses.pop(pos)
        return positons
        
        '''
        fitnesses.sort()
        
        
        
        eq_population = list(self.equations)
        eq_population.sort(key=get_fitness)
        
        rand = np.random.randint

        #mod_population = [Equation(14,rand(1,15)) for i in range(num)]        
        
        for i in range(num):
            eq_population[i] = Equation(14, rand(1,15))
        
        self.equations = np.array(eq_population)
        '''
        
            
        
    
                
        
            
    
    def evolve(self):
        """
        Puts the functions together.
        This could be done in the driver file instead.
        """
        
        #How many generations will there be:
        avg_fit = self.calculate_population_fitness()
        for i in range(200):
            print("Average Fitness: ",avg_fit)
            self.mutation(0.35)                     # I'm setting a really high mutation rate because crossover is not implemented, so this is the only way for the population to change
            avg_fit = self.calculate_population_fitness()
            self.set_new_generation()
            
        # Get the best individual:
        best = 0,0
        for i in range(self.size):
            (current_best,pos) = self.equations[i].get_fitness(),i
            if current_best[0] > best[0]:
                best = current_best
        
        best_equation = self.equations[best[1]]
        
        str_eq = decoder_14Bits(best_equation)
        print("Winner: ",str_eq)
        return str_eq
        
        
        
        
        
        
        
        
        