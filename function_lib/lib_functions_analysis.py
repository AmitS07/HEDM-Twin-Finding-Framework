'''
Date: Saturday Sept. 26 2023
Author: Amit, ACDM Lab
Descripsion: This library contaions all the general functions

'''

import pandas as pd
import numpy as np
from math import pi, cos, sin, acos, sqrt, tan
import random
from scipy.spatial.transform import Rotation as ROT
from collections import Counter

##########################################################################
#%% Part 0: Some general functions common to all
##########################################################################
## 0.4 "Volume weightage" calculation
def volume_weightage(r1,r2):
    # Input: gives the radius of two grains, & 
    # Note: "spherical" volume assumed 
    v1 = (4/3)*pi*r1**3
    v2 = (4/3)*pi*r2**3
    w1 = v1/(v1+v2)
    w2 = v2/(v1+v2)
    w = [w1,w2]
    return w

## 0.5 "non_unique_elements" 
def non_unique_elements(matched_grains_list):
    # Input: matched_grains_list = list from which it has to be estimated
    # Output: list of repeated elements of the given data
    element_counts = Counter(matched_grains_list) # This will count each unique element once, but if repeated element is there then it will be 2
    non_unique_items = []
    for element, count in element_counts.items():
        if count>1:
            non_unique_items.append(element)
    return non_unique_items

## 0.7 to calculate "euclidean_dist"
def euclidean_dist(cord1, cord2):
    '''
    Description: This is to Find the EUCLIDEAN-DISTANCE between given two coordinate, 
    Input: Give the coordinate of two points, either as in 'array' or 'list' form
    Output: Euclidean-dist
    Note: Make sure Numpy is imported as np
    '''
    euclid = np.linalg.norm(np.array(cord2)-np.array(cord1))
    return euclid

##########################################################################
#%% 2. Function related to calculate "confidence" : from Karthik
##########################################################################
## 2.1
def confidence(dist, tol):
    # dist = distance b/w the points
    if dist < tol:
        # assuming linear relation
        CI = 1 - (dist/tol)
    else:
        CI = 0
    return CI
## 2.2
def cumulative_confidence(CIs, ws=[1]):
    if len(ws) == 1:
        ws = []
        _ws = ws[0]
        for _ in CIs:
            ws.append(_ws)
    CI = 0
    for w, _CI in zip(ws, CIs):
        CI += w*_CI
    return CI

## 2.3 
def centroid_confidence(centroid1, centroid2, tol, domain="rectangular"):
    if domain =="rectangular":
        CIs = []
        for a,b,t in zip(centroid1,centroid2,tol):
            dist = abs(b-a)
            CI_xyz = confidence(dist,t)
            CIs.append(CI_xyz)
        # CI = CIs[0]*CIs[1]*CIs[2] # for 3D
        CI = CIs[0]*CIs[1] # for 2D
    else: 
        # Assume spherical
        rad = max(tol)
        # Euclidean  dist
        euci_dist = np.linalg.norm(np.array(centroid2)-np.array(centroid1))
        if euci_dist<rad:
            CI = 1-(euci_dist/rad)
        else:
            CI = 0
    return CI


#############################################################################
#%% 3. Function and text code to check whether a function is present or not
###############################################################################
#%% 3.1 Function
def check_function_occurrences(directory, function_name):
    '''
    Descripson: This is to check the how many places a given function occurs
                in a set of python functions
    '''
    import os
    import re
    occurrences = []  # To store tuples of (file_path, line_number)

    pattern = rf"def\s+{function_name}\s*\("  # Regex pattern to match function definitions

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):  # Only process Python files
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    for line_number, line in enumerate(f, start=1):
                        if re.search(pattern, line):  # Search for the function definition
                            occurrences.append((file_path, line_number))
    return occurrences
''' 
#%% 4.2 Test/ Example
directory = '/home/asingh38/codes/nsf_twinning_2022'
function_name = "rot2mat"
results = check_function_occurrences(directory, function_name)

# Output the results
if results:
    print(f"The function '{function_name}' is found in the following locations:")
    for file_path, line_number in results:
        print(f"File: {file_path}, Line: {line_number}")
else:
    print(f"The function '{function_name}' was not found in any files.")
'''
#############################################################################
#%% 4 Dictionary related functions
###############################################################################
#4.1 Read Dictionary saved as 'csv'
def read_csv_dictionary(csv_file_path):
    # Required libraries
    import csv
    import ast
    # defining 'dictionary'
    dict= {}
    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for rows in reader:
            key = int(rows[0]) # The first column is the key
            values = [ast.literal_eval(v) for v in rows[1:]] # The remaining columns are the values
            dict[key] = values
    return dict

#4.2 To get the dictionary Values based on Key
def get_dictionary_values(dictionary, key):
    if key in dictionary:
        values = dictionary[key][0] # Extract the list of lists
        value1 = np.array(values[0],dtype=float)
        value2 = np.array(values[1],dtype=float)
        return value1, value2
    else: 
        return None, None
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-6: Function to read-write txt data 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%6.1 To read the data
def read_numeric_txt(file_path):
    '''
    Input: just give the path to the file, which are being read
    Note: This function is very specific, especially deisgn to read numerical text data
    '''
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split line and convert each element to int or float
            row = []
            for elem in line.strip().split():
                try:
                    row.append(int(elem))  # Try converting to integer first
                except ValueError:
                    try:
                        row.append(float(elem))  # Fall back to float if needed
                    except ValueError:
                        row.append(elem)  # Keep as string if non-numeric
            data.append(row)
    return data

##%6.2 To write the data
def write_numeric_txt(data, file_path):
    with open(file_path, 'w') as f:
        # Check if data is flat (e.g., [0, 1, 0])
        if not isinstance(data[0], (list, np.ndarray)):
            for item in data:
                f.write(f"{item}\n")  # One line per element  # Write single line
        else:
            for row in data:
                f.write(" ".join(map(str, row)) + "\n")  # Multi-line

