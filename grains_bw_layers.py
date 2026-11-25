'''
Created: Saturday Sept. 16 2023
Updated: 14th August 2025, for Sample-3, just added to import the librabries
Author: Amit 
Description: Code for grain-matching layer-wise
Note:1. This code was written for the HEDM data-sets which has fixed increment of load-steps (e.g. in sample#2, 25) So if load-steps incremented differently..
        then "while-loop" has to be modified to "for-loop" or something else based on the incremented load-steps
     2. Caution: while selecting "Z-offset" for each type of sample, For sample#1, Z-Offset has to added as-
                Sample#1:: Layer-1# Z+0, Layer-2: Z+(Z_offset}, Layer-3: Z+(2*Z_offset)
                Sample#2:: Layer-1# Z+0, Layer-2: Z+(Z_offset}, Layer-3: Z+(2*Z_offset)
                Sample#3:: Layer-1: Z+Z_offset, Layer-2: Z+0, Layer-3: Z+(2*Z_offset)
'''
print(f"***"*20)
print(f"Wait! Be careful in accounting Z_offsets bw layers \n Each sample was measured differently")
print(f"***"*20)
# %%
# Importing Libraries
import sys
import os
import glob
from turtle import shape
import pandas as pd
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from math import pi, cos, sin, acos, sqrt
import random
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
#
#########################################################################################################
#########################################################################################################
# %%

#%%%%%%%%%%%% Functions  %%%%%%%%%%%%%%%
def grainIDs_map(newID,oldID):
    grain_map_dict = {}
    grain_map_dict['newID'] = newID
    grain_map_dict['oldID'] = oldID
    return grain_map_dict
#########################
def write_json(dict_data, json_file, indent=4):
    with open(json_file, 'w') as fp:
        json.dump(dict_data, fp, indent=indent)
    return None
#########################
def trim_drop_IDs_map(trimmedID,droppedID):
    trimmed_grainIDs_dic = {}
    trimmed_grainIDs_dic['Trimmed_grainsID'] = trimmedID
    trimmed_grainIDs_dic['Dropped_grainsID'] = droppedID
    return trimmed_grainIDs_dic

#%%%%%%%%%%%% time tic    %%%%%%%%%%%%%%%
# tic = time.time()
#########################################################################################################
########################### Part:1 Reading file and Selecting layer-wise data ###########################
#########################################################################################################
#0. Defining folder path and listing all the files
pth = '/home/asingh38/projects/nsf_twinning_2022'
sam_n = eval(input("Enter sample number:")) #3 
z_offset = eval(input("Enter Z-offset:")) #200
load_range = list(map(int, input(f"Enter all loads (space seprated): ").split()))
print(f"Load-Range:", load_range)
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load: {load_s}")
print(f"Final load: {load_f}")
# Entering grain matching: flitering criteria
tol_euclid = eval(input("Enter euclidean tolerance:")) #50 tolerance in euclidean distance
tol_misori = eval(input("Enter misori tolerance:")) #1, tolerance in misori
wt_misori= eval(input("Enter misori weightage:")) #0.5, # weightage of misori
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for load_c in load_range:
    flsp  = glob.glob(f'{pth}/raw_data/sample_{sam_n}/parquet_files/sam{sam_n}_{load_c}N_L*.parquet')
    flsp.sort()
    # print(flsp)
    for flpth in flsp:
        fn = os.path.basename(flpth)
        splt_fls = fn.split('_')
        layer = splt_fls[2][-1]
        if layer == '1':
            df1_all = pd.read_parquet(flpth) #For smaple#1, 2, and Layer-1: No change to Z-offset
            df1_all['Z'] += z_offset # For Sample#3, need to add the Z-offset
            print(f"#Grains in sample {sam_n} Load {load_c} Layer {layer} with z-offset {z_offset}:", len(df1_all))
        elif layer == '2':
            df2_all = pd.read_parquet(flpth) #For smaple#3, and Layer-2, No change to Z-offset
            # df2_all['Z'] += z_offset #For smaple#1, 2, and Layer-2: subtarct Z-offset
            print(f"#Grains in sample {sam_n} Load {load_c} Layer {layer} 1with z-offset {z_offset}:",len(df2_all))
        else: 
            df3_all = pd.read_parquet(flpth)
            # df3_all['Z'] -= z_offset*2 #For smaple#1, 2, and Layer-3: subtract 2*Z-offset
            df3_all['Z'] -= z_offset #For smaple#3, and Layer-3: subtract Z-offset
            print(f"#Grains in sample {sam_n} Load {load_c} Layer {layer} with z-offset {z_offset}:",len(df3_all))
    print(35*"--")
################################################################################################################################
########################### Part:2 Finding out matched grains based on "Orientation" and " Position" ###########################
################################################################################################################################
    # ## Tic
    tic = time.time()
    ############ Calculating the matched grains between layer1 and layer2 = matched_grains_l12 ############
    ####################################################################################################### 
    df_matched_grains_l12 = grains_matching_layerwise(df1_all,df2_all,tol_euclid,tol_misori,wt_misori)
    df_matched_grains_l12.to_csv(f'{pth}/raw_data/sample_{sam_n}/matched_grains_layerwise/sam{sam_n}_{load_c}N_layer1_2_matched_grains_Z{z_offset}_misori{tol_misori}_eucid{tol_euclid}_CI{wt_misori}_O321.csv', index = True)

    # ### Time tooks for merging layer1-2
    toc1 = time.time()
    elapsed1 = (toc1-tic)/60
    print("Time elapsed(m) for matching grains bw layer1-2:", elapsed1)
    print(f"Total Number of Matched Grains b/w Layer 1-2 Load {load_c} with z_offset {z_offset}:",len(df_matched_grains_l12))

    ################################################################################################################################################
    ########################### Testing for any double matching grains b/w one grain of layer-1 to two grains of layer-2 ###########################
    df1_modified,df_matched_modi_l12 = check_and_remove_double_match_grains(df_matched_grains_l12,df1_all)

    #%%%%%%%%%%%% Calculating "Average_CIs" %%%%%%%%%%%%%%%%
    avg_com_CIs_l12 = np.average(df_matched_grains_l12['Comulative_CI'])
    print("Average comulative CIs b/w layer 1-2:",avg_com_CIs_l12)
    print(50*"-")

    ################################# Mearging the data of matched grains between two layers (l1 and l2) ##################################
    # df1_reduced and df2_reduced are datafiles after removing the mathced grains from the actual raw data based on volume weighatges 
    df1_reduced, df2_reduced =  merged_grains_two_layers(df1_modified,df2_all,df_matched_modi_l12)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ############ Calculating the matched grains between layer2 and layer3 = matched_grains_l23 ############
    ####################################################################################################### 
    ## Note: To matched the grains b/w between layer2 and layer3, the updated layer2 data i.e. "df2_reduced" is considered 
    df_matched_grains_l23 = grains_matching_layerwise(df2_reduced,df3_all,tol_euclid,tol_misori,wt_misori)
    df_matched_grains_l23.to_csv(f'{pth}/raw_data/sample_{sam_n}/matched_grains_layerwise/sam{sam_n}_{load_c}N_layer2_3_matched_grains_Z{z_offset}_misori{tol_misori}_eucid{tol_euclid}_CI{wt_misori}_O321.csv', index = True)

    ### Time tooks for merging layer2-3
    toc2 = time.time()
    elapsed2 = (toc2-toc1)/60
    print("Time elapsed(m) for matching grains bw layer2-3:", elapsed2)
    print(f"Total Number of Matched Grains b/w Layer 2-3 Load {load_c} with z_offset {z_offset}:",len(df_matched_grains_l23))

    ################################################################################################################################################
    ########################### Testing for any double matching grains b/w one grain of layer-2 to two grains of layer-3 ###########################
    df2_modified,df_matched_modi_l23 = check_and_remove_double_match_grains(df_matched_grains_l23,df2_reduced) # df2_reduced is the current raw_data fro df2 after merging two layers

    #%%%%%%%%%%%% Calculating "Average_CIs" %%%%%%%%%%%%%%%%
    avg_com_CIs_l23 = np.average(df_matched_grains_l23['Comulative_CI'])
    print("Average comulative CIs b/w layer 2-3:",avg_com_CIs_l23)
    print(50*"-")

    ################################# Mearging the data of matched grains between two layers (l1 and l2) ##################################
    # df2_final and df3_reduced are datafiles after removing the mathced grains from the actual raw data based on volume weighatges 
    df2_final, df3_reduced =  merged_grains_two_layers(df2_modified,df3_all,df_matched_modi_l23) 

########################################################################################################################################
################################# Part:3 Mearging the data of matched grains between all three layers ##################################
########################################################################################################################################
    df_l123 = pd.concat([df1_reduced,df2_final,df3_reduced])
    ## Saving as parquet file
    df_l123.to_parquet(f'{pth}/raw_data/sample_{sam_n}/parquet_files/sam{sam_n}_{load_c}N_layered_grains_O321.parquet', index = True)
    # ## Saving as .csv file
    # df_l123.to_csv(f'{pth}/raw_data/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N.csv', sep = "\t", index = True) # index = True, add "new_GrainID"
    print(f"Total number of the grains in sample {sam_n}-load {load_c}:",len(df_l123))
    print("Total time elapsed(m) for merging layers:", elapsed1+elapsed2)
    print(25*"===")





