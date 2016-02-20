#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      berejnoibejaranoc
#
# Created:     19/01/2016
# Copyright:   (c) berejnoibejaranoc 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()


def _gradCheck_helper(cost_fun, epsilon = 10E-4):
    """Helper"""
    thetaPlus = cost_fun + epsilon              #cost_fun should be a numpy array containing all the errors for the given layer, I think...
    thetaMinus = cost_fun - epsilon

    numerical_gradient = (thetaPlus - thetaMinus) / (2*epsilon)

    return numerical_gradient

def gradientCheking(gradients, cost_fun, epsilon = 10E-4, tolerance_difference = 10E-4):
    """Checks if the gradients computed by backpropagation are good enough"""

    num_gradient = _gradCheck_helper(cost_fun,epsilon)

    # Doing some weird stuff down here, do not mind me...
    plus = (gradients < (num_gradient + tolerance_difference))
    minus = (gradients > (num_gradient - tolerance_difference))
    checked = np.logical_and(plus,minus).astype(int)
    return checked




