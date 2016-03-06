# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 00:32:13 2016

@author: andresberejnoi
"""
import numpy as np

def decoder_14Bits(gene=np.array([0, 0,0,0,0,0, 0,0,0, 0, 0,0,0,0]) ):
    """
    A gene decoder. It takes the gene as a binary and turns it into the corresponding
    string.
    gene: The 14-bit binary information for a particular gene. The data structure used is a 
            numpy array because the size of the gene will not change.
            The gene is arranged in the following way: [sign_bit coefficeint_integer_part coefficient_decimal_part x_indicator exponent_for_x]
            The following are the pieces, one by one:
            - sign_bit: 1 or 0. 1 indicates number is negative and 0 indicates it is positive
            - coefficient: The number that goes in front of X, ex : in 3x, the coefficient is 3. Here is is separated into the integer part and fractional part:
                * coefficient_integer_part: 5 bits, so biggest integer part can be '11111'= 31
                * coefficient_decimal_part: 3 bits, in the way a fraction is represented in binary: first bit is 1/2, next one is 1/4, then 1/8, and so on.
                * example of coefficient as a whole, the binary 11100.001 will be 28 for the integer part and fractional = 0+0+1/8 = 1/8, so number will be 28.125
            - x_indicator: 1 or 0; it tells whether the variablex is part of this equation or not. 1 means yes, and 0 is no
            - exponent_for_x: 4 bits; encode the exponent that x will be raised to. However, in order to allow for negative exponets, the bits are shifted 7 values to the left.
                                Therefore, '0000' will mean -7, '0001' = -6, ..., '0111'=0, ..., '1111' = 8
                                (For now, only integer exponents are allowed)
    """
    #text = "+("                 # The terms in the equation will all be added to the next in order to make processing easier. If the value is negative, the minus sign will be inside the parenthesis
    text = ""
    #checking gene is acceptable:
    try:
        assert (len(gene)==14)
    except: 
        raise DecoderError("Gene is not 14 bits long. Using 14-bit decoder")
    
    #if the coefficient is 0, we will just return 0, without checking the values
    if np.sum(gene[1:9]) == 0:                 #Only add values of coefficient. If x indicator is false, then we do not care about exponents, and if the coefficient is zero, then nothing matters anymore
        text += '+0'
        return text
    
    # We extract the information of each area in the gene to convert to string
    if gene[0]:                # #if the sign bit is 1, then the number is negative and we put a '-' in front, otherwise, put a '+'
        text += "-"   
    else:
        text += "+"
    # =========================================================================        
    # Finding the coefficient value
    #   Integer part first
    coeff_int = gene[1:6][::-1]                     #a slice view of the gene at the integer part, reversing the bits to  have least significant as the first item
    indices = np.arange(len(coeff_int))             # creates a new arrange with index values for the same size as 
    int_part = np.sum(coeff_int * (2**indices))     # first, it creates an array of the values of 2 raised to the powers of the index positons, then the bit values are multiplied to get an array of decimal values for each binary position, finally the numbers are added together to get final base 10 number 
        
    #   Fractional part:
    coeff_frac = gene[6:9][::-1]                    # gets the slice of values for the fractional part of the coefficient
    frac_part = np.sum(coeff_frac * (1 / (2**indices[1:len(coeff_frac)+1]) ) )        # re-using indices to avoid creating a new numpy array for this. The problem is that makes the code less clear
    
    if int_part + frac_part != 1:                   # we will only add the coefficient into the string if it is different than 1. If it is one, we do not need to add it.
        text += str(int_part)    
        text += str(frac_part)[1:]                      # the fractional part is added to the text, making sure to only add the digits after the '.' (that is what the slicing is doing, and the '.' is included)
    #==========================================================================
    # Determing if x is present:
    if gene[9]:                    #if the x indicator bit is 1, we will append an x to the string
        #Here we deal with the exponent, only when x is present
        value_shift = 7             # the value by which the binary values are shifted
        exp_bits = gene[10:14][::-1]                # this process is going to repeat the steps from above to calculate a binary integer. It will probably be better to write that algorithm into a helper function and call it 2 times
        exp_part = np.sum(exp_bits * (2**indices[:len(exp_bits)]) )             #same process as above. Also, reusing indices. it could be made a class variable to not calculate it every time this function is called 
        exp_part = exp_part - value_shift                                       #substract the shifted amount to get the actual expected base 10 exponent
        
        if (exp_part == 0):                 # if the exponent is zero, then x will always be 1, and is not necessary to include
            return text
        
        text += 'x'
        if (exp_part !=1):                  # only include the exponent if exponent is not 1 
            text += """^({0})""".format(str(exp_part))                              # concatenates the value of the exponent, surrounded by parenthesis to make it easier to read

    #text += ")"
    return text
    
def to_bin(num):
    """
    Takes a number in base 10 and converts it to binary.
    num: a Python base 10 number. It should be a positive integer
    """    
    try:
        assert(type(num)==int)
    except AssertionError:
        print ("The number provided must be a positive integer")

            
        
def encoder_14Bits(eq_term = "0"):
    """
    Does the opposite of the decoder. Given a string in the form of an equation term, it will return an array with the encoded information.
    eq_term: a Python string; it's the equation term we want to encode into one gene. Note that it is a single term in the equation, because
            each term will become a different gene. Ex: the equation x^(2) + 3x has two terms, so eq_term would be 'x^(2)' or '3x', but not both together.
            Furthermore, to simplify this function, it will be required that the exponent be surrounded by parenthesis after the '^'. Ex: 'x^(2) is good, but 'x^2' will raise an exception.
            Finally, the coefficient must go in front of x, with no symbols in between ('*' or others). Ex: '3x^(2)' is accepted, but '3*x^(2)' or 'x^(2)*3' will raise an exception. 
    """    
    
    # Setting variables for the pieces:
    sign_bit = []
    co_int = []            #bits for the integer coefficient
    co_frac =[]                # bits for the fractional coefficient, there will be 3, but they will be appended later 
    x_flag = []
    e_bits = []              # bits for the exponent part
    
    coeff_string = ""    
    
    try:
        float(eq_term)              # This will trigger an exception if the equation contains anything other than numbers
        
        x_flag = [0]
        e_bits = [0,0,0,0]
        
        # if the value is 0, we return the value for zero, without calculation anything
        if float(eq_term)==0.0:
            return np.array([0, 0,0,0,0,0, 0,0,0, 0, 0,0,0,0])

        coeff_string = eq_term
        '''
        #we check if the string has a fractional part
        if '.' in eq_term:
            int_str,frac_str = eq_term.split('.')                           #separater integer part from fractional
        else: 
            int_str = eq_term
                
        if int(int_str) < 0:                        #if the number is negative, the sign bit is 1, otherwise it will be 0
            sign_bit = [1]
        else: sign_bit = [0]
        '''
        
    except ValueError:              #we get to this point if the convertion to float from above failed because the string contains more than just numbers
        
        x_flag = [1]                # we will assume that if we get to this branch, then x must be present
        
        if eq_term.lower().endswith('x'):                                       #If there is nothing after x, then exponent is assumed to be 1
            e_bits = [0,0,0,1]
            coeff_string = eq_term[:-1]                                         # we can ignore the x at the end      
        else:
            
            coeff_string, exponent = eq_term.lower().split("x^")                # Separating the string into 2 pieces, the coefficient (together with the sign), and the exponent
            #handle the exponent bits inside this else clause:
            exponent = int(exponent[1:len(exponent)-1])                              # This removes the parenthesis of the exponent, and makes it into an integer
            if (exponent < -7) or (exponent > 8):
                raise EncoderError("""Exponent is not in correct range.\n Range:[-7,8]\tGiven: {0}""".format(exponent))
            
            exponent += 7                                                           # the +7 is to account for the shift value in the exponent for binary interpretation           
            e_bits = [int(bit) for bit in bin(exponent)[2:]]                        # bin is a built-in function. It returns the binary of an integer as a string, including the leading '0b' part, so the slicing is removing that part of the string
            
            #Now make sure that there are exactly 4 bits in e_bits:
            if len(e_bits) > 4:
                e_bits = e_bits[:4]                 # gets only the first 4 bits
            else:
                reversed_exp = e_bits[::-1]
                while len(reversed_exp) < 4:
                    print("reversed_exp: ",reversed_exp)
                    reversed_exp.append(0)
                    
                e_bits = reversed_exp[::-1]
                print("e_bits:",e_bits)
        #========================End of Except clause==============================
   
   #Check that value is in correct range
    #print("Coeff_string:",coeff_string)
    if (float(coeff_string) > 31.875) or (float(coeff_string) < -31.875):
        raise EncoderError("""Coefficient value provided is not in acceptable range.\n Range:[-31.875,31.875]\tGiven: {0}""".format(coeff_string))
    # Check whether the number is negative or positive
    if (float(coeff_string) >= 0):
        sign_bit = [0]
    else: 
        sign_bit = [1]
        coeff_string = coeff_string[1:]                     # after we used the minus sign, we can remove it from the string because it is no longer needed
    
    #Split the coefficient into its parts (integer, fractional):
    if '.' in coeff_string:
        int_str,frac_str = coeff_string.split('.')                           #separater integer part from fractional
        
        frac_bits = []
        fraction = float("0."+frac_str)
        
        #Convert fraction into binary
        while fraction != 0:                         
            fraction = fraction*2
            #print (fraction)
            frac_bits.append(int(fraction))
            fraction = fraction % 1
        co_frac = frac_bits[:3]                           # We can only store the first 3 fractional bits
        #we check to make sure there are 3 fractional bits; if there are less, we pad with zeros
        while len(co_frac) < 3:
            co_frac.append(0)
    else: 
        int_str = coeff_string
        co_frac = [0,0,0]
    
    co_int = [int(bit) for bit in bin(int(int_str))[2:]]
    
    #making sure there are exactly 5 bits in the integer part of the coefficient:
    if len(co_int) > 5:
        raise EncoderError("There are more than 5 bits in the integer coefficient")
    else:
        reversed_int = co_int[::-1]
        while len(reversed_int) < 5:
            print("reversed_int: ",reversed_int)
            reversed_int.append(0)
            
        co_int = reversed_int[::-1]
        print("e_bits:",co_int)
    
    
        
    

    

    print("bits: ",sign_bit, co_int, co_frac, x_flag, e_bits)
    gene = sign_bit + co_int + co_frac + x_flag + e_bits                # putting the pieces together.
    
    #make sure the output gene has 14-bits exactly:
    assert(len(gene)==14)
    
    return np.array(gene)                           #returns numpy array of the gene


#
#   Exception objects
#
class GAError(Exception):
    """The basic template for GA-related exceptions in this file"""
    def __init__(self):
        self.msg = "Error occurred"
        
    def __str__(self):
        return self.msg
        
    __repr__=__str__
        
        
class DecoderError(Exception):
    def __init__(self, msg="Gene does not have correct format"):
        self.msg = msg
        
class EncoderError(Exception):
    def __init__(self, msg="Expression does not have right format"):
        self.msg = msg

class Equation (object):
    """
    The methods beloging to an individual member of the population.
    """
    def __init__(self):
        pass



class EquationGA (object):
    def __init__(self):
        pass
    
    
def gene_collection():
    """Just a momentary location to put some encoded genes to make it easier to test on the fly"""
    
    zero =      np.array([0,  0,0,0,0,0,  0,0,0,  0,  0,0,0,0])                  #several configurations can give 0, as long as the coefficient bits are all zeros
    two =       np.array([0,  0,0,0,1,0,  0,0,0,  0,  1,0,0,1])
    x_squared = np.array([0,  0,0,0,0,1,  0,0,0,  1,  1,0,0,1])                 # '+x^(2)'
    other =     np.array([1,  0,1,1,1,0,  0,0,1,  1,  1,0,0,1])                 # '-14.5x^(2)'
    
encoder_14Bits('0.1x^(2)')    