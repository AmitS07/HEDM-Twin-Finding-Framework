''' 
Date: Tuesday, 24th March 2026
Author: Amit, ACDM Lab, The University of Alabama, USA, Email: asingh38@crimson.ua.edu, amitdravid@outlook.com
Descripsion: This library contains the functions that are related to crystal-plasticity mechanics
'''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-0: Importing python libraries 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from lib_mathematics import *

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-1: Continumm_mechanics functions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def yield_point(strain,stress, yield_offset=0.002):
    E = stress[1]/strain[1]
    # stress_y_offset = E*strain -E*yield_offset
    stress_y_offset = [(E * i) - (E * yield_offset) for i in strain]
    for i in range(len(strain)):
        if stress_y_offset[i]>stress[i]:
            n1 = i
            n0 = n1-1
            break
    # The slope and intercept for offset line
    m1 = E
    c1 = -E*yield_offset
    # The slope and intercept for line form stress-strain
    m2 = (stress[n1]-stress[n0])/(strain[n1]-strain[n0])
    c2 = stress[n0]-(m2*strain[n0])
    # yeild point
    y_strain = (c2-c1)/(m1-m2)
    y_stress = m1*y_strain+c1
    values = [y_strain, y_stress, stress_y_offset]
    return values

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-2: Basic general function related to vigot notation and equivalent stress/strain
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%2.1 To convert a matrix to voigt array
def symmetric_matrix_to_voigt_array(matrix, format):
    '''
    Input: The symmetric square matrix of n*n dimensions
    Output: vector of the elements from the upper triangular matrix
    Note: If one want to have the Voigt notation otherthan "strain", then
          at type any string otherthan "strain"
    Link: https://en.wikipedia.org/wiki/Voigt_notation
    '''
    matrix = np.array(matrix) # making sure that matrix is an array
    #---------------------------------------------------
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")
    #---------------------------------------------------
    n = matrix.shape[0]
    diagonal = np.diag(matrix) # Extract diagonal elements
    upper_triangular = [] # Extract upper triangular elements (excluding diagonal)
    for i in range(n):
        for j in range(i+1, n):
                if format == "strain":
                    upper_triangular.append(2*matrix[i, j]) ## for strain
                else: 
                    upper_triangular.append(matrix[i, j]) ## for stress
    # Combine diagonal and upper triangular elements
    voigt_vector = np.concatenate([diagonal, upper_triangular[::-1]])
    voigt_matrix = voigt_vector.reshape(-1, 1) # Reshape into a column matrix
    return voigt_matrix

#%% 2.2: To convert voigt-array to matrix
def voigt_array_to_symmetric_matrix(voigt_array, format):
    '''
    Input: Voigt-array, tell forrmat i.e conversion is for "Stress" or "Strain"
    Output: Matrix form of given Voigt notation
    Note: This function does convert the "Voigt" form of stress/strain to
        "Matrix" form
    '''
    # For Stress to convert from voigt to matrix
    n = len(voigt_array) # length of the array
    d = int(n/2) # Dimension of the matrx
    diag_mat = np.zeros([d,d]) #Digonal-part of the matrix
    for i in range(d):
        diag_mat[i,i] = voigt_array[i]
    if format == "strain": ## For strain
        trig_mat = np.zeros([d,d]) #Upper-Trigonal-part of the matrix
        k = d 
        for i in range(d):
            for j in range(i+1,d):
                trig_mat[i,j] = 1/2*(voigt_array[k+2])
                k-=1
        #%% Full-matrix
        sym_mat = diag_mat+trig_mat+trig_mat.T
    else: ## for stress
        trig_mat = np.zeros([d,d]) #Upper-Trigonal-part of the matrix
        k = d 
        for i in range(d):
            for j in range(i+1,d):
                trig_mat[i,j] = voigt_array[k+2]
                k-=1
        #%% Full-matrix
        sym_mat = diag_mat+trig_mat+trig_mat.T 
    return sym_mat

#%%2.3: To calculate equivalent_stress 
def equivalent_stress(stress_matrix):
    '''
    Input: stress_matrix should be a 3*3 matrix, defined in array form
    Output: Equivalent stress
    Note: It required "dobledot" product function. Also, it is same as if we use the all componets of stress and use standard formula, Mtheod#2
    '''
    ##Volumentric stress component
    sigma11, sigma22, sigma33 = stress_matrix[0,0], stress_matrix[1,1], stress_matrix[2,2]
    ## Devaitoric stress component
    sigma12, sigma13, sigma23 = stress_matrix[0,1], stress_matrix[0,2], stress_matrix[1,2]
    ## Equivalent stress calculation: Method-1
    trace = sigma11+sigma22+sigma33 # Trace of the matrix
    I = np.array([[1,0,0], 
                  [0,1,0],
                  [0,0,1]]) # Identity matrix
    hydrostatic_stress = (1/3)*trace*I # Hydrostaric part of stress state
    deviatoric_stress = stress_matrix-hydrostatic_stress # Deviatoric part of stress-state
    stress_eq = np.sqrt((3/2)*doubledot(deviatoric_stress,deviatoric_stress)) #(3/2) factor ensures that, under uniaxial loading, the von Mises equivalent stress equals the applied stress at yield.
    ## Equivalent stress calculation: Method-2
    # stress_eq = np.sqrt(0.5*(((sigma11-sigma22)**2)+((sigma22-sigma33)**2)+((sigma33-sigma11)**2)+ 6*((sigma12**2)+(sigma13**2)+(sigma23**2))))
    ##%% Note: Both method are same
    return stress_eq

#%%2.4: To calculate equivalent_strain 
def equivalent_strain(strain_matrix):
    '''
    Input: strain_matrix should be a 3*3 matrix, defined in array form
    Output: Equivalent strain
    Note: strain_eq from both method are different, but finally we are considering Method-1 (FEPX one)
    This function required function "doubledot"
    '''
    e11, e22, e33 = strain_matrix[0,0], strain_matrix[1,1], strain_matrix[2,2]
    e12, e13, e23 = strain_matrix[0,1], strain_matrix[0,2], strain_matrix[1,2]
    ## Method-1: Ref: FEPX
    I = np.array([[1,0,0],
                  [0,1,0], 
                  [0,0,1]])
    trace = e11 + e22 + e33
    hydrostatic = I*(trace)/3
    deviatoric = strain_matrix - hydrostatic
    strain_eq = np.sqrt((2/3)*doubledot(deviatoric,deviatoric))
    # ## Method-2, Accounted Poission ration, Ref: Ansys (https://innovationspace.ansys.com/courses/wp-content/uploads/sites/5/2020/05/Lesson-5-Evaluate-Strain-Results.pdf)
    # nu = 0.33 # poisson's ratio
    # strain_eq_a = (1/(1+nu))*np.sqrt((1/2)*((e11-e22)**2 + (e22-e33)**2 +(e33-e11)**2 + 6*((e12**2) + (e13**2) + (e23**2))))
    # ## Method-3, same as Method#1
    # m  =trace/3
    # strain_eq_j = np.sqrt((2/3)*((e11-m)**2 + (e22-m)**2 +(e33-m)**2 + 2*((e12**2) + (e13**2) + (e23**2))))
    return strain_eq