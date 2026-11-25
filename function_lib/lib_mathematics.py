'''
Date: 30th October 2025
Author: Amit Singh, ACDM lab
Description: Functions related mathematics- Such as-
                Linear-algebra, set theory

'''
##########################################################################
#%% 0. Importing Libaries
##########################################################################
#0. Importing libraries from in-bulid python
import pandas as pd
import numpy as np
from math import pi, cos, sin, acos, sqrt, tan

##########################################################################
#%% Part-1. Function related to sets 
##########################################################################
#3.1 To find out: Common items
def set_intersection(data1, data2):
    common_elements = list(set(data1).intersection(data2))
    return common_elements

#3.2 To find out: not common items
def set_difference(data1, data2):
    diff_12 = list(set(data1) -set(data2))
    diff_21 = list(set(data2) -set(data1))
    diff_elements  = diff_12+ diff_21 # Appending two list
    return diff_elements
#3.3 To find the common elements between multiple arrays/lists
def intersection_elements_multiple_arrays(*arrays):
    '''
    This function is to select out the common elements between multiple data-sets
    Input: All the arrays/list from which, one want to find out the intersection
    Output: list of final common elements among given 
    Example: 
    int = intersect_multiple_arrays(array1, array2, array3)
    '''
    if not arrays:
        print(f"Opps! array not provided")
        return []
    sets = [set(arr) for arr in arrays]
    result = sets[0].intersection(*sets[1:])
    return list(result)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-2: Functions related to basic linear-algebra, vectors and matrix
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##2.1: To normalize vectors
def normalize_vector(vect, magnitude=False):
    # from Ezra
    value= 0
    final = vect
    for i in vect:
        value+=(i**2)
    mag=sqrt(value)
    for i in range(len(vect)):
        final[i] = final[i]/mag
    if magnitude:
        return [final,mag]
    else:
        return final
##2.2 To  convert list/array to n*n matrix
def list_to_matrix_nxn(list):
    '''
    Input: List of variables with length of n*n values, elment has to be in following order
            [M11, M12, M13, M21, M22, M23, M31, M32, M33]
    Output: Matrix of variables with n*n dimensions
    Note: This function is same as "rot2mat" function in "lib_orientations_symmetries"
    '''
    n = int(len(list)**0.5) # dinesion of the matrix 
    matrix = np.zeros((n,n))
    counter = 0 # to index the vaiables in the list
    for i in range(n):
        for j in range(n):
            matrix[i,j] = list[counter]
            counter += 1
    return matrix
##2.3 To calculate matrix dot product i.e. double-dot product
def doubledot(a,b):
    ''' 
    It is a "doubledot" product function is dot product between two matrix
    link: https://stemandmusic.in/maths/mvt-algebra/matrixDDP.php
    '''
    ## Checking matrix_symmetricity
    if a.shape != b.shape[1]:
        ab = np.sum(np.multiply(a, b))
    else:
        ab = np.trace(np.matmul(a,b.T))
    return ab
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-3: Functions related to basic linear-algebra, vectors and matrix
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%3.1 Function for basic airthematic
def greatest_common_divisor(a, b):
    while b:  
        a, b = b, a % b  # Update 'a' to 'b' and 'b' to 'a % b'
    return a  