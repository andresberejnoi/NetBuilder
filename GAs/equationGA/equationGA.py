# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 00:32:13 2016

@author: andresberejnoi
"""

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
    text = "+("                 # The terms in the equation will all be added to the next in order to make processing easier. If the value is negative, the minus sign will be inside the parenthesis
    
    #checking gene is acceptable:
    try:
        assert (len(gene)==14)
    except: 
        raise GeneError("Gene is not 14 bits long. Using 14-bit decoder")
    
    #if the gene is 0, we will just return 0, without checking the values
    if np.sum(gene[1:9]) == 0:                 #Only add values of coefficient. If x indicator is false, then we do not care about exponents, and if the coefficient is zero, then nothing matters anymore
        text += '0)'
        return text
    
    # We extract the information of each area in the gene to convert to string
    if gene[0]:                # #if the sign bit is 1, then the number is negative and we put a '-' in front, otherwise, it is positive and we do not put anything
        text += "-"   
    # =========================================================================        
    # Finding the coefficient value
    #   Integer part first
    coeff_int = gene[1:6][::-1]                     #a slice view of the gene at the integer part, reversing the bits to  have least significant as the first item
    indices = np.arange(len(coeff_int))             # creates a new arrange with index values for the same size as 
    int_part = np.sum(coeff_int * (2**indices))     # first, it creates an array of the values of 2 raised to the powers of the index positons, then the bit values are multiplied to get an array of decimal values for each binary position, finally the numbers are added together to get final base 10 number 
    text += str(int_part)
    
    #   Fractional part:
    coeff_frac = gene[6:9][::-1]                    # gets the slice of values for the fractional part of the coefficient
    frac_part = np.sum(coeff_frac * (1 / (2**indices[1:len(coeff_frac)+1]) ) )        # re-using indices to avoid creating a new numpy array for this. The problem is that makes the code less clear
    text += str(frac_part)[1:]                      # the fractional part is added to the text, making sure to only add the digits after the '.' (that is what the slicing is doing, and the '.' is included)
    #==========================================================================
    # Determing if x is present:
    if gene[9]:                    #if the x indicator bit is 1, we will append an x to the string
        text += 'x'
        #Here we deal with the exponent, only when x is present
        value_shift = 7             # the value by which the binary values are shifted
        exp_bits = gene[10:14][::-1]                # this process is going to repeat the steps from above to calculate a binary integer. It will probably be better to write that algorithm into a helper function and call it 2 times
        exp_part = np.sum(exp_bits * (2**indices[:len(exp_bits)]) )             #same process as above. Also, reusing indices. it could be made a class variable to not calculate it every time this function is called 
        exp_part = exp_part - value_shift                                       #substract the shifted amount to get the actual expected base 10 exponent
        text += """^({0})""".format(str(exp_part))                              # concatenates the value of the exponent, surrounded by parenthesis to make it easier to read

    text += ")"
    return text

class GeneError(Exception):
    def __init__(self, msg="Gene is not correct"):
        self.msg = msg
    
    def __str__(self):
        return self.msg
        
    __repr__=__str__
        


class Equation (object):
    """
    The methods beloging to an individual member of the population.
    """
    def __init__(self):
        pass



class EquationGA (object):
    def __init__(self):
        pass
    
