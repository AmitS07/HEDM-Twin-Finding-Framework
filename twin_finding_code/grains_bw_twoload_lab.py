'''
Date: 20th August 2025, Wednesday
Author: Amit, ACDM Lab, The University of Alabama, USA
Decription: This scripts does the calculation of misorientations and Euclidean distance, for each grain parirs
            between two consecutive load-steps, in Part-1. In part-2, based on teh given filtering criteria of 
            misorienation and euclidean distance, it will find the matched-grains bwteen those tow consecutive load-steps and save it. 
'''
# %%
# Code for Grain_indexing with out using MIDAS#
# Importing Libraries
import os
import sys
# from turtle import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, acos, sqrt
import time

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %%
## This is added to call the functions libraries
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022')
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# importing manually defined functions 
from lib_orientations_symmetries import *
from lib_functions_analysis import *
from lib_grains_matching import *
from lib_grains_matching_mp import *
#%%%%%%%%%%%% time tic    %%%%%%%%%%%%%%%
# tic = time.time()
#########################################################################################################
########################### Part:1 Reading file and Selecting load-wise data ###########################
#########################################################################################################
#0. Defining folder path and listing all the files
pth = '/home/asingh38/projects/nsf_twinning_2022'
sam_n = eval(input(f"Enter the sample number: "))
load_range = list(map(int, input(f"Enter all loads, seperatedby space: ").split()))
print(f"Load-Range:", load_range)
load_s = load_range[0]
load_f = load_range[-1]
print(f"First load for analysis: {load_s} N")
print(f"Last load for analysis: {load_f} N")
print(22*"***")

##################################################################################################################################################################
########## Part-2: This script is similar to MIDAS version of calulation of "misori" and "eculidean" bw  two consecutive load-steps ##############
##################################################################################################################################################################
''' 
weight = eval(input(f"Enter filtering weight for misori:"))
for i in range(0, len(load_range),1):
    # Reading the (i)th load step data 
    load_i =load_range[i]
    df1_all = pd.read_csv(f'{pth}/raw_data/sample_{sam_n}/csv_files/sam{sam_n}_{load_i}N_grains.csv', skiprows = 1, sep = "\t").set_index("%new_GrainID")
    print(f"#Grains in sample {sam_n} Load {load_i} :", len(df1_all))
    # Reading the (i+1)th load step data
    load_ii = load_range[i+1]
    df2_all  = pd.read_csv(f'{pth}/raw_data/sample_{sam_n}/csv_files/sam{sam_n}_{load_ii}N_grains.csv', skiprows = 1, sep = "\t").set_index("%new_GrainID")
    print(f"#Grains in sample {sam_n} Load {load_ii} :", len(df2_all))
    tic = time.time()
    ############2.1: "misorientation" and "euclidean dist" bw two-consecutive load steps, & save as a matrix ############
    #################################################################################################################### 
    df_grains_l12 = map_misori_euclideanDist_bw_twoLoad(df1_all, df2_all, weight)
    df_grains_load12 = df_grains_l12.sort_values(by='minAngle')
    df_grains_load12.to_csv(f'{pth}/raw_data/sample_{sam_n}/matched_grains_loadwise/matrix_misori_euclid/sam{sam_n}_{load_i}_{load_ii}N_misori_eucid_acme.csv', index=False)
    toc1 = time.time()
    elapsed1 = (toc1-tic)/60
    print(f"Time elapsed(m) for matching grains bw sam{sam_n}_{load_i}_{load_ii}N:", round(elapsed1,3))
    print(25*"===")
    #%% to lose the loop
    if load_ii == load_f:
        break
##Note:: This section of the code does directly generate the misorientation and euclidean-dist values for matched
# pair of the grains. During analysis equal weight given to misorienation and euclidean-dist. But few grain pairs results are 
# are not matching with the output of the MIDAS code "grains_bw_twoload_midas.py". Which might be due to that MIDAS code account
# weight to euclidean dist differently.
'''
################################################################################################################################
########################## MATRIX version:Alternative version but more roboust, hence more time taking ##############
########################### Part:3  "misorientation" and "euclidean dist" bw two-consecutive load steps, & save as a matrix ###########################
##### Note:: Need to parallelize it
################################################################################################################################
# '''
print(f"!!MATRIX approach to calculate the misori and euclidean!!")
print("---"*22)
print(f"Caution: Check number of processor available \n Don't give more than 50% of your machine cores")
print("---"*22)
n_cores = eval(input(f"Enter number of processor available:"))
for i in range(0, len(load_range),1):
    tic = time.time()# Calculation start time 
    # Reading the (i)th load step data 
    load_i =load_range[i]
    df1_all = pd.read_csv(f'{pth}/raw_data/sample_{sam_n}/csv_files/sam{sam_n}_{load_i}N_grains.csv', skiprows = 1, sep = "\t").set_index("%new_GrainID")
    print(f"#Grains in sample {sam_n} Load {load_i} :", len(df1_all))
    # Reading the (i+1)th load step data
    load_ii = load_range[i+1]
    df2_all  = pd.read_csv(f'{pth}/raw_data/sample_{sam_n}/csv_files/sam{sam_n}_{load_ii}N_grains.csv', skiprows = 1, sep = "\t").set_index("%new_GrainID")
    print(f"#Grains in sample {sam_n} Load {load_ii} :", len(df2_all))
    #%%------------------------------------------------------------------------------
    ############2.1: "misorientation" and "euclidean dist" bw two-consecutive load steps, & save as a matrix ############
    # matrix_grains_ls12 = calc_misori_euclideanDist_bw_twoLoad(df1_all,df2_all)
    ############2.2(using multiprocessing): Misori and euclid calculation bw two load-steps in Matrix
    matrix_grains_ls12 = calc_misori_euclideanDist_bw_twoLoad_mp(df1_all,df2_all, n_cores)
    #%%------------------------------------------------------------------------------
    toc1 = time.time()# calculation finish time
    elapsed1 = (toc1-tic)/60
    print(f"Time elapsed(m) for matching grains bw sam{sam_n}_{load_i}_{load_ii}N:", round(elapsed1,3))
    # Saving the Matrix
    matrix_grains_ls12.to_csv(f'{pth}/raw_data/sample_{sam_n}/matched_grains_loadwise/matrix_misori_euclid/sam{sam_n}_{load_i}_{load_ii}N_misori_eucid_matrix.csv')
    # ### Time tooks for merging layer1-2
    toc2 = time.time()# calculation finish time
    elapsed2 = (toc2-toc1)/60
    print(f"Time took to save the grain match matrix sam{sam_n}_{load_i}_{load_ii}N:", round(elapsed2,3))
    print(25*"===")
    #%% to lose the loop
    if load_ii == load_f:
        break
print(f"Matrix of Misori and Euclid-dist. stored!!")
print(22*"***")
# ''' 
########## Part-3 .1:: Reading matrix and extracting GrainIDs bw two consecutive load-steps based on given "misori" and "euclidean distt" criteria ##############
'''
print(f"Reading matrix for misori and euclidean dist!")
# %% Defining local function
def str_to_array(s):
    import ast
    # Safely convert string representation of list to Python list
    py_list = ast.literal_eval(s)
    return np.array(py_list)
#% Grains matching: flitering criteria
misori_tol = eval(input(f"Enter misori. tolerance:"))
euclid_tol = eval(input(f"Enter Euclid tolerance:"))
for i in range(0, len(load_range),1):
    # Calculation end
    tic = time.time() # Calculation begin time
    load_i = load_range[i]
    load_ii = load_range[i+1]
    ## Reading Matrix
    df_mat = pd.read_csv(f"{pth}/raw_data/sample_{sam_n}/matched_grains_loadwise/matrix_misori_euclid/sam{sam_n}_{load_i}_{load_ii}N_misori_eucid_matrix.csv", index_col=0)
    ## Selecting Grain IDS
    gIDs1 = df_mat.index.tolist()
    gIDs2 = df_mat.columns.tolist()
    ## Selecting grainIDs  based on filtered "misori" and "euclidean distt" tolerance
    common_grains_ls12 = []
    for ii, gID1 in enumerate(gIDs1):
        for jj, gID2 in enumerate(gIDs2):
            cell_str = df_mat.iloc[ii, jj]
            arr = str_to_array(cell_str)
            misori = arr[0]
            euclid = arr[1]
            if misori<= misori_tol and euclid<=euclid_tol:
                common_grains_ls12.append([gID1,gID2,misori,euclid])
    common_grains = pd.DataFrame(common_grains_ls12)
    #%%------------------------------------------------------------------------------
    #%  Checking for number of matched grains
    if len(common_grains)==0:
        print(f"Opps! There is not any matched grains, \n Check the tolerances!!")
        break
    #%%------------------------------------------------------------------------------
    common_grains.columns = ["OrigIDState1", "OrigIDState2", "minAngle", "EuclideanDistt"]
    df12_common_grains = common_grains.sort_values(by = 'minAngle') 
    df12_common_grains.to_csv(f'{pth}/raw_data/sample_{sam_n}/matched_grains_loadwise/matrix_misori_euclid/sam{sam_n}_{load_i}_{load_ii}N_misori_eucid.csv', index = False)
    #Calculation begin
    toc = time.time() #Calculation finish time
    print(f"Time elapsed(m) for matching grains bw sam{sam_n}_{load_i}_{load_ii}N:", round((toc-tic)/60,3))
    # to Stop the for-loop, when two-loads are used. 
    if load_ii == load_f:
        break
# '''

