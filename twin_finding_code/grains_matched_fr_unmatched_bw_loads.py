'''
Date: 9th July 2024
Author: Amit Singh, ACDM lab
Description: This script is for selecting out the mathced grains from "a" load steps to nth load step. \\
             the "a" load should be increasing and tending towards "nth" load step incremnetally, and \\
             finding  out the matched grains and un-matched grains

             INPUT: The list or dataframe of all the unmatched/raw grains from 2nd laod steps to nth load steps
             OUTPUT: List of matched and un-matched grains across all load steps
Concept: It is to only consider the "twins" which are developed during the deformation, not twins which are \\
         present from the beginnings.
Note: For this there is no need to run the MIDAS. Hence, this script is same as the "grains_bw_loads_midas.py" after Part:1 of that script.
'''
# Note: For this there is no need to run the MIDAS. Hence, this script is same as the "grains_bw_loads_midas.py" after Part:1 of that script.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++ Part:0 Importing required libaries and functions ++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import pandas as pd
import sys
import os
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This is added to call the functions libraries
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022')
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from lib_grains_matching_load_midas import *

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++ Part:1 Defining path and giving fitting parameters and tolerances ++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
sam_n = eval(input("Enter the sample number:")) #%% Sample number
load_range = list(map(int, input(f"Enter all loads (space seprated) being analyzed: ").split()))
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")
z_offset = eval(input("Enter the Z_offset(0):")) # =0
beamheight = eval(input("Enter the Beamthickness:")) # =1200 for sample-2 and 600 for sample#3 ## It is beamheight, i.e z-offset bw each load steps
twinAngle = eval(input("Twin-angle(0{not considering twin} else 85)(in deg):")) # =0 
tol_misori = eval(input("Misori-tolerance(in deg):")) #1 in degree
tol_euclid = eval(input("Enter Euclidian-dist tolerance(um):")) #50 in um
print(25*"+++")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++ Part:2 Importing matched garins data frame after MIDAS run ++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
load_first = eval(input("Enter first applied load:")) #%%  It should always we the very 1st load i.e. 0N
load_final = eval(input("Enter last applied load:")) #%% It is the final tested load
# Defining load
# Reading grainIDs of unmatched grains from 0-300 load or 0-n load steps:
unmatched_df = pd.read_csv(f'{pth}/sample_{sam_n}/unmatched_grains_loadwise/final_unmatched_grainsIDs_loadwise_{load_first}_{load_final}.csv',sep='\t')

if load_s != load_range[-2] and len(load_range)>2: # This condition is applied for all load steps, except last two load
    print(f"---------------Working data-set bw load-range {load_s}-{load_f}------------")
    common_grainsId_L3 = {}
    count = 1
    for i in range(0, len(load_range),1): #This make sure, we are doing the analysis upto 250N laod steps, since this part is going to analysis 2-load step ahead
        load_i = load_range[i]
        load_ii = load_range[i+1]
        load_iii = load_range[i+2]
        #%%%%%%%%%%%%%%%%%% Post analysis of matched grains after MIDAS run %%%%%%%%%%%%%%%%%%%%%%#
        #---------------------------------------------------------------------------------------------
        # Filtering unmatched grains-ID from the MIDAS data: Load-step k-(k+1)
        # Selecting out the unmatched-grains-ID based on given load-step = k
        unmatched_IDs_LS1 = unmatched_df[f'Load_{load_i}'].dropna()
        # Reading MIDAS data for a load step of 1-2
        df_L12 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
        unmatched_L12 = df_L12[df_L12['OrigIDState1'].isin(unmatched_IDs_LS1)] # This is to remove the grain's IDs which are not in the list of unmatched-IDs
        #---------------------------------------------------------------------------------------------
        # Filtering unmatched grains-ID from the MIDAS data: Load-step (k+1)-(k+2)
        # Selecting out the unmatched-grains-ID based on given load-step = k+1
        unmatched_IDs_LS2 = unmatched_df[f'Load_{load_ii}'].dropna()
        # Reading MIDAS data for a load step of 2-3
        df_L23 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_ii}_{load_iii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
        unmatched_L23 = df_L23[df_L23['OrigIDState1'].isin(unmatched_IDs_LS2)]
        #-------------------------------------------------------------------------------------------
        #%%%%%%%%%%%%%%%%%%%%% Case-1: By filtering grains based on tolerance %%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%%%%%%%%%%%%%%% Checking for similar grains IDs L123 %%%%%%%%%%%%%%%%%%%%%%%
        #%% Matched grains bw two-load steps: L12
        match_grainsID_L1,match_grainsID_L2 = filter_matched_grainsIDs(unmatched_L12, tol_misori, tol_euclid)
        print(f"# Matched grains bw Load {load_i}_{load_ii}:", len(match_grainsID_L2))
        #%% Matched grains bw two-load steps: L23
        match_grainsID_L2_n,match_grainsID_L3 = filter_matched_grainsIDs(unmatched_L23, tol_misori, tol_euclid)
        print(f"# Matched grains bw Load {load_ii}_{load_iii}:", len(match_grainsID_L3))

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if count == 1:
            common_grainsId_L3[f'Load_{load_iii}'] = intersectioned_grains(match_grainsID_L2,match_grainsID_L2_n,match_grainsID_L3)
            # # To select common grains from L2
            # common_grains_L3[f'Load_{load+icrmt}'] = intersectioned_grains(common_grains_L3[f'Load_{load+icrmt*2}'],match_grainsID_L2_n,match_grainsID_L2_n)  
            # print(f"Matched grains IDs in L{load+icrmt*2} corresponding to L123: {len(common_grains_L3[f'Load_{load+icrmt}'])}")  # print load L2
            print(f"Matched grains IDs in load-{load_iii} corresponding to L123: {len(common_grainsId_L3[f'Load_{load_iii}'])}") # to print load L3
            count += 1
        else:
            common_grainsId_L3[f'Load_{load_iii}'] = intersectioned_grains(common_grainsId_L3[f'Load_{load_ii}'],match_grainsID_L2_n,match_grainsID_L3)
            print(f"Matched grains IDs in load-{load_iii} corresponding to L123: {len(common_grainsId_L3[f'Load_{load_iii}'])}")

            # Save these "Common_grains_IDs" itetatively"? Which will help later to comparegainst final matched grains
        print(25*"+++")
        # To stop the  loop
        if load_iii == load_f:
            break
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Note: This whole part is added just to save a matched-grains bw load 275-300, since this is the last pair of the data set and Using MIDAS data, with gievn criteria a single 
#% set of matched grains pair was made.
else: 
    print(f"---------------Working on data-set bw just last two loads {load_s} & {load_f}------------")
    load_i = load_range[0]
    load_ii = load_range[1]
    common_grainsId_L3 = {}
    #%%%%%%%%%%%%%%%%%% Post analysis of matched grains after MIDAS run %%%%%%%%%%%%%%%%%%%%%%#
    #---------------------------------------------------------------------------------------------
    # Filtering unmatched grains-ID from the MIDAS data: Load-step k-(k+1)
    # Selecting out the unmatched-grains-ID based on given load-step = k
    unmatched_IDs_LS1 = unmatched_df[f'Load_{load_ii}'].dropna()
    # Reading MIDAS data for a load step of 1-2
    df_L12 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
    unmatched_L12 = df_L12[df_L12['OrigIDState2'].isin(unmatched_IDs_LS1)] # This is to remove the grain's IDs which are not in the list of unmatched-IDs
    #---------------------------------------------------------------------------------------------
    # Filtering unmatched grains-ID from the MIDAS data: Load-step (k+1)-(k+2)
    # Selecting out the unmatched-grains-ID based on given load-step = k+1
    unmatched_IDs_LS2 = unmatched_df[f'Load_{load_ii}'].dropna()
    # Reading MIDAS data for a load step of 2-3
    df_L23 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
    unmatched_L23 = df_L23[df_L23['OrigIDState2'].isin(unmatched_IDs_LS2)] # This is- to read the grans-IDs the from the MIDAS file for laod 300, therfore, "OrigIDState2" used
    #---------------------------------------------------------------------------------------------
    #%%%%%%%%%%%%%%%%%%%%% Case-1: By filtering grains based on tolerance %%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%% Checking for similar grains IDs L123 %%%%%%%%%%%%%%%%%%%%%%%
    #%% Matched grains bw two-load steps: L12
    match_grainsID_L1,match_grainsID_L2 = filter_matched_grainsIDs(unmatched_L12, tol_misori, tol_euclid)
    print(f"# Matched grains bw Load {load_i}_{load_ii}:", len(match_grainsID_L2))
    #%% Matched grains bw two-load steps: L23
    match_grainsID_L3,match_grainsID_L2_n = filter_matched_grainsIDs(unmatched_L23, tol_misori, tol_euclid)
    print(f"# Matched grains bw Load {load_i}_{load_ii}:", len(match_grainsID_L3))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    common_grainsId_L3[f'Load_{load_ii}'] = intersectioned_grains(match_grainsID_L1,match_grainsID_L3,match_grainsID_L2_n)
    # # To select common grains from L2
    # common_grains_L3[f'Load_{load+icrmt}'] = intersectioned_grains(common_grains_L3[f'Load_{load+icrmt*2}'],match_grainsID_L2_n,match_grainsID_L2_n)  
    # print(f"Matched grains IDs in L{load+icrmt*2} corresponding to L123: {len(common_grains_L3[f'Load_{load+icrmt}'])}")  # print load L2
    print(f"Matched grains IDs in load-{load_ii} corresponding to L123: {len(common_grainsId_L3[f'Load_{load_ii}'])}") # to print load L3
    # Save these "Common_grains_IDs" itetatively"? Which will help later to comparegainst final matched grains
    print(25*"+++")
    
## Save Dictionary file to dataframe
common_grainsId_loadwise = pd.DataFrame.from_dict(common_grainsId_L3, orient='index').T #'orient='index', makes dictionary's keys to become the index of df, and values of dictionary makes column of df & T - Transpose
common_grainsId_loadwise.to_csv(f'{pth}/sample_{sam_n}/matched_grains_fm_unmatched_loadwise/common_grainIds_loadwise_{load_s}_{load_f}.csv', sep='\t',index=False)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++ Part:2  Finding common grains across all data based on final unmatched dataset ++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# ''' 
# 1.1 Loading the list/dataframe of the unmatched grains for all load steps
print(25*"===")
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")
print(f"--"*20)

common_grainsId_finalload = common_grainsId_loadwise[f'Load_{load_f}'].tolist()
print(f"Common Grains IDs on Final load steps {load_f}:{len(common_grainsId_finalload)}")

print("#+++ Part:3  Finding common grains across all data based on final dataset +++#")

final_common_grainsIds = {}
count = 1
for i in range(len(load_range)-1, 0, -1): # For decreasing order of loads
    load_i = load_range[i-1]
    load_ii = load_range[i]
    #%%%%%%%%%%%%%%%%%% Post analysis of matched grains after MIDAS run %%%%%%%%%%%%%%%%%%%%%%#
    #%% Reading file of matched grains
    df_L12 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')

    #%%%%%%%%%%%%%%% Checking for similar grains IDs L123 %%%%%%%%%%%%%%%%%%%%%%%
    #%% Matched grains bw two-load steps: L12
    match_grainsID_L1,match_grainsID_L2 = filter_matched_grainsIDs(df_L12, tol_misori, tol_euclid)
    print(f"# Matched grains bw Load {load_i}-{load_ii}:", len(match_grainsID_L2))

    if count ==1:
        final_common_grainsIds[f'Load_{load_ii}'] =  common_grainsId_finalload
        print(f"Common grains IDs corresponding to final load L{load_ii}: {len(final_common_grainsIds[f'Load_{load_ii}'])}")

        final_common_grainsIds[f'Load_{load_i}'] = common_matched_grains(final_common_grainsIds[f'Load_{load_ii}'],match_grainsID_L2,match_grainsID_L1)
        print(f"Common grains IDs corresponding to final load L{load_i}: {len(final_common_grainsIds[f'Load_{load_i}'])}")
        count +=1
    else:
        final_common_grainsIds[f'Load_{load_i}'] = common_matched_grains(final_common_grainsIds[f'Load_{load_ii}'],match_grainsID_L2,match_grainsID_L1)
        print(f"Common grains IDs corresponding to final load L{load_i}: {len(final_common_grainsIds[f'Load_{load_i}'])}")
    print(25*"+++")

## Save Dictionary file to dataframe
final_common_grainsId_loadwise = pd.DataFrame.from_dict(final_common_grainsIds, orient='index').T #'orient='index', makes dictionary's keys to become the index of df, and values of dictionary makes column of df & T - Transpose
final_common_grainsId_loadwise.to_csv(f'{pth}/sample_{sam_n}/matched_grains_fm_unmatched_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv', sep='\t',index=False)

# '''
