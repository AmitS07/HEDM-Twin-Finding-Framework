'''
Date: 15th May 2024
Author: Amit Singh, ACDM (Advanced Computational Deformation Mechanics) Lab 
Descripson: This code will selectc out unmatched grains(potential twin) of the final matched grains 
            across the load steps (indeed, after merging layer-wise grains)
'''
# importing libaries
# from in-build python
import pandas as pd
import sys
import os
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This is added to just call the functions libraries
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022')
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing library
from lib_grains_matching_load_midas import *

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++ Part:0 Defining path and and sample's directory +++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
sam_n = eval(input("Enter the sample number:")) #%% Sample number
load_range = list(map(int, input(f"Enter all loads (space seprated) which you want to analyze: ").split()))
print(f"Load-Range:", load_range)
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")
print(f"---"*20)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++ Part:1 Reading matched grains load-wise data (after matching layer wise) for load across all load steps +++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#1.1 Reading saved "Final_common_grainsId_loadwise" file
load_first = eval(input("Enter first applied load:")) #%%  It should always we the very 1st load i.e. 0N
load_final = eval(input("Enter last applied load:")) #%% It is the final tested load
final_matched_grainsIDs  = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_{load_first}_{load_final}.csv', sep='\t', index_col=False)
final_matched_grainsIDs  = final_matched_grainsIDs.dropna()
print(f"Total matched grains from load range {load_first}-{load_final}:",len(final_matched_grainsIDs))
print(f"---"*20)
#%%=========================================================================
#1.2 Looping over for all-load steps
final_unmatched_grainsIDs = {}
for load_c in load_range:
    final_matched_grainsID = final_matched_grainsIDs[f"Load_{load_c}"] # Note: Never give same as
    #1.2.1 Reading Final "matched-grains" data-frame layer-wise i.e. for each load step one layer 
    df_grains = pd.read_csv(f'{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv', skiprows=1, sep='\t', index_col=False) # df_load_modi_id =data with modified id i.e deleting 1st digit as sample number and intermediate "0"
    df_grains_ids = df_grains.iloc[:,0] ## Selecting all grainsIDs of a load
    print(f"Total #matched grains at Load-{load_c}:", len(df_grains_ids))
    #1.2.2 Finding final matched grains on comparison to load-wise to layer-wise matched data 
    final_unmatched_grainsID = set_difference(df_grains_ids,final_matched_grainsID)
    final_unmatched_grainsIDs[f"Load_{load_c}"] = final_unmatched_grainsID
    print(f"Total #unmatched grains for load {load_c}:",len(final_unmatched_grainsID))
    #*************************************************************************************************
    ## Verify: whether is there any unmatched grains present in mathced grains
    missed_grainsIDs = set_intersection(final_unmatched_grainsID,final_matched_grainsID)
    if len(missed_grainsIDs)>0:
        print(f"Opps! These unmatched grain should not present in matched grains!")
    #***************************************************************************************************
    print(f"**"*25)
final_unmatched_grains_IDs = pd.DataFrame.from_dict(final_unmatched_grainsIDs, orient='index').T 
final_unmatched_grains_IDs.to_csv(f'{pth}/sample_{sam_n}/unmatched_grains_loadwise/final_unmatched_grainsIDs_loadwise_{load_s}_{load_f}.csv', sep='\t',index=False)
