''' 
Date: Tuesday, 24th March 2026
Author: Amit, ACDM Lab, The University of Alabama, USA, Email: asingh38@crimson.ua.edu, amitdravid@outlook.com
Descripsion: This library contains the functions that are related to crystal-plasticity mechanics
'''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-0: Importing python libraries 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# import numpy as np


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