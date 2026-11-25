'''
Created: 29th Feb 2024
Author: Amit Singh, ACDM (Advanced Computational Deformation Mechanics) Lab 
Descripson: This code will do the grains-matching between load steps, after running from MIDAS
            (i.e. after having misorientation and eucliden-dist data), and then
            merged the common matched grains between the two based on Ids
Modified date: 21st August 2025, for data set which was not taken at constant load increment.
'''
# importing libaries
# from in-build python
import os
import glob
import sys
import pandas as pd
import numpy as np
import time 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This is added to call the functions libraries
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022')
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing library: maunalu defined
from lib_grains_matching_load_midas import *

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++ Part:1 Reading file and Running MIDAS for load across all load steps +++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
sam_n = eval(input("Enter the sample number:")) #%% Sample number
load_range = list(map(int, input(f"Enter all loads, seperatedby space: ").split()))
print(f"Load-Range:", load_range)
#----------------- Part:1.1 Parameters used in running MIDAS for load across all load steps --------------#
z_offset = eval(input("Enter the Z_offset:")) # =0
beamheight = eval(input("Enter the Beamheight:")) #beamheight, i.e z-offset bw each load steps, sample#3 = 600, sample#2 = 1200
twinAngle = eval(input("Enter Twin-angle(0{not considering twin} else 85)(in deg):")) # =0 
tol_misori = eval(input("Enter Misori-tolerance(in deg):")) #1 in degree for sample-3, 0.5 for samaple2
tol_euclid = eval(input("Enter Euclidian-dist tolerance(um):")) #50 in um
load_s = load_range[0]
load_f = load_range[-1]
print(f"Start applied-load: {load_s}")
print(f"Final applied-load: {load_f}")
print(f"---"*20)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++ Part:1 Importing matched garins data frame after MIDAS run ++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# ''' 
if load_s != load_range[-2] and len(load_range)>2: # This condition is applied for all load steps, except last two load
    print(f"---------------Working data-set bw load-range {load_s}-{load_f}------------")
    count = 1
    common_grainsId_L3 = {}
    for i in range(0, len(load_range),1):
        load_i = load_range[i]
        load_ii = load_range[i+1]
        load_iii = load_range[i+2]
    #     #%%%%%%%%%%%%%%%%%% Post analysis of matched grains after MIDAS run %%%%%%%%%%%%%%%%%%%%%%#
        #%% Reading file of matched grains
        df_matched_L12 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
        df_matched_L23 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_ii}_{load_iii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
        #%%%%%%%%%%%%%%%%%%%%% Case-1: By filtering grains based on tolerance %%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%%%%%%%%%%%%%%% Checking for similar grains IDs L123 %%%%%%%%%%%%%%%%%%%%%%%
        #%% Matched grains bw two-load steps: L12
        match_grainsID_L1,match_grainsID_L2 = filter_matched_grainsIDs(df_matched_L12, tol_misori, tol_euclid)
        print(f"# Matched grains bw Load {load_i}-{load_ii}:", len(match_grainsID_L2))
        #%% Matched grains bw two-load steps: L23
        match_grainsID_L2_n,match_grainsID_L3 = filter_matched_grainsIDs(df_matched_L23, tol_misori, tol_euclid)
        print(f"# Matched grains bw Load {load_ii}-{load_iii}:", len(match_grainsID_L3))
        # common_grains_L3 = finding_common_matched_IDs('123', match_grainsID_L2_n,match_grainsID_L2,match_grainsID_L3)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if count == 1:
            common_grainsId_L3[f'Load_{load_iii}'] = intersectioned_grains(match_grainsID_L2,match_grainsID_L2_n,match_grainsID_L3)
            # # To select common grains from L2
            # common_grains_L3[f'Load_{load+icrmt}'] = intersectioned_grains(common_grains_L3[f'Load_{load+icrmt*2}'],match_grainsID_L2_n,match_grainsID_L2_n)  
            # print(f"Matched grains IDs in L{load+icrmt*2} corresponding to L123: {len(common_grains_L3[f'Load_{load+icrmt}'])}")  # print load L2
            print(f"Matched grains IDs in L{load_iii} corresponding to L123: {len(common_grainsId_L3[f'Load_{load_iii}'])}") # to print load L3
            count += 1
        else:
            common_grainsId_L3[f'Load_{load_iii}'] = intersectioned_grains(common_grainsId_L3[f'Load_{load_ii}'],match_grainsID_L2_n,match_grainsID_L3)
            print(f"Matched grains IDs in L{load_iii} corresponding: {len(common_grainsId_L3[f'Load_{load_iii}'])}")
            # Save these "Common_grains_IDs" itetatively"? Which will help later to comparegainst final matched grains
        print(25*"+++")
        
        # To stop the  loop
        if load_iii == load_f:
            break
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Note: This whole part is added just to save a matched-grains bw last two loads, which actually generated directly using MIDAS. Hence there is no need to run sepretly 
#% set of matched grains pair was made.
else: 
    print(f"---------------Working on data-set bw just last two loads {load_s} & {load_f}------------")
    load_i = load_range[0]
    load_ii = load_range[1]
    common_grainsId_L3 = {}
    #%% Reading file of matched grains
    df_matched_L12 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
    df_matched_L23 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
    #%%%%%%%%%%%%%%% Checking for similar grains IDs L123 %%%%%%%%%%%%%%%%%%%%%%%
    #%% Matched grains bw two-load steps: L12
    match_grainsID_L1,match_grainsID_L2 = filter_matched_grainsIDs(df_matched_L12, tol_misori, tol_euclid)
    print(f"# Matched grains bw Load {load_i}-{load_ii}:", len(match_grainsID_L2))
    #%% Matched grains bw two-load steps: L23
    match_grainsID_L3,match_grainsID_L2_n = filter_matched_grainsIDs(df_matched_L23, tol_misori, tol_euclid) # The matched list is swicthed
    print(f"# Matched grains bw Load {load_i}-{load_ii}:", len(match_grainsID_L3))
    common_grainsId_L3[f'Load_{load_ii}'] = intersectioned_grains(match_grainsID_L1,match_grainsID_L3,match_grainsID_L2_n)
    print(f"Matched grains IDs in L{load_ii} corresponding to L123: {len(common_grainsId_L3[f'Load_{load_ii}'])}") # to print load L3
    print(25*"+++")
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Save Dictionary file to dataframe
common_grainsId_loadwise = pd.DataFrame.from_dict(common_grainsId_L3, orient='index').T #'orient='index', makes dictionary's keys to become the index of df, and values of dictionary makes column of df & T - Transpose
common_grainsId_loadwise.to_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/common_grainIds_loadwise_{load_s}_{load_f}.csv', sep='\t',index=False)
# '''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++ Part:2  Finding common grains across all data based on final dataset ++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# ''' 
print(25*"===")
print(f"Starting load: {load_s}")
print(f"Final load: {load_f}")
print(f"--"*20)
common_grainsId_finalload = common_grainsId_loadwise[f'Load_{load_f}'].tolist()
# print(common_grainsId_finalload)
print(f"Common Grains IDs on Final load steps {load_f}:{len(common_grainsId_finalload)}")

print("#+++ Part:3  Finding common grains across all data based on final dataset +++#")

final_common_grainsIds = {}
count = 1
# while load_final>load_s:
for i in range(len(load_range)-1, 0, -1):
    load_i = load_range[i-1]
    load_ii = load_range[i]
    #%%%%%%%%%%%%%%%%%% Post analysis of matched grains after MIDAS run %%%%%%%%%%%%%%%%%%%%%%#
    #%% Reading file of matched grains
    df_matched_L12 = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamheight}.csv',sep='\t')
    
    #%%%%%%%%%%%%%%% Checking for similar grains IDs L123, By filtering grains based on tolerance %%%%%%%%%%%%%%%%%%%%%%%
    #%% Matched grains bw two-load steps: L12
    match_grainsID_L1,match_grainsID_L2 = filter_matched_grainsIDs(df_matched_L12, tol_misori, tol_euclid)
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
final_common_grainsId_loadwise.to_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv', sep='\t',index=False)

# '''






