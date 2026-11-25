'''
Date: 10th July 2024
Author: Amit Singh, ACDM Lab
Description: This script does compare the matched grains for a given load \\
             to find the "new" grains which could be potential "twins" by ensuring that \\
             one twin should not have more than one parent-grain
Output: Dictionary of twin-IDs and Parent-grain-IDs for the key being as the load-step. This is the oder, not PG and twin
'''
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++ Part:0 Importing required libaries and functions ++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import pandas as pd
import numpy as np
import sys
import os
import time
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This is added to call the functions libraries
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022') # Path to the function libraries
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from lib_grains_matching import twin_parent_pair_finding_at_load
from lib_orientations_symmetries import calc_theoretical_misori_quat_hcp
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++ Part:1 Theoretical quternion calculation of parent-twin misorientation ++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
tw_misori_angle = float(input("Enter twin misori angle(in degree): "))
tw_misori_axis = np.array(input("Enter axis of misori (4-indices) separated by spaces: ").split(),dtype=int)
lattice_a  = 2.95 # in Angostrom for HCP Titanium
covera = 1.588 # c/a ratio of HCP lattice for HCP Ti
q_th = calc_theoretical_misori_quat_hcp(tw_misori_angle, tw_misori_axis, lattice_a, covera, crystal_align_to="a2||y", degree = True)
print(f"Quternion of misori(angle/axis): \n {q_th}")
print(25*"***")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#%% Part:2 Finding matched grains(from unmatched grains) as a "potential twins" while comapring to final matched grains 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## Defining sample varibales and parameters
sam_n = eval(input("Enter the sample number:")) #%% Sample number
load_range = list(map(int, input(f"Enter all loads (space seprated) being analyzed: ").split()))
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")
# twin_angle = eval(input("Enter Twin-angle(deg, depend on twin-type):"))#
twin_tol = eval(input("Enter the twin-misori angle tol.(deg):")) # twin_tol = 2.5
pg_rad_tol = eval(input("Enter parent-grain radius tol.(um):")) # pg_rad_tol = 25
avg_rad_factor = eval(input("Enter avg. radius factor (1, if not any):")) # keep = 1; for not considering the radius-factor, else as calculated e.g. 2.19
approach = input("Enter type of approach('wr' if rad_pg>rad_tw, else 'wor'):") # wr(with radius) i.e by accounting rad_pg>rad_tw, wor(with out radius) =  with out accounting rad_pg>rad_tw
consider = input(f"Enter type of analysis('all' or 'unique'):") # enter 'unique', if you want to consider no repated matched grans
print(25*"***")
#%% Defining path
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'

# ''' 
#%2.1 Loading final matched grainIDs i.e list of grains-match across all load steps
final_matched_grainsIDs = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_0_320.csv", sep = "\t",index_col = False)
#%2.2 Loading matched-grains from the unmatched grains of incremntal loading, for this we are not considering the garins/potential twins which are present at load-0
matched_grainsIDs_um = pd.read_csv(f'{pth}/sample_{sam_n}/matched_grains_fm_unmatched_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv', sep='\t', index_col=False)
#%%*******************************************To iterate over load-steps***************************************************
tw_pg_dict = {} # To store the found twins and corresponding parent grains
scale_factor = {}
## looping over all the loads
for load_c in load_range:
    tic = time.time()
    ##1.3 Reading final matched grain IDs and matched-IDs from unmatched grain IDs
    # Reading load-wise final-matched grains IDs as parent grains
    parent_IDs_load = final_matched_grainsIDs[f"Load_{load_c}"]
    pg_IDs = parent_IDs_load.dropna()
    # Reading load-wise matched grains from unmathced-grain list as potential-twins (ptw)
    ptwin_IDs_load = matched_grainsIDs_um[f"Load_{load_c}"]
    ptw_IDs = ptwin_IDs_load.dropna()
    #%% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #% Importing final considered grains data
    df_load = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv",skiprows=1, sep = "\t")
    df_load = df_load.set_index(["%new_GrainID"])
    twin_pg_IDs, scale_factor = twin_parent_pair_finding_at_load(scale_factor, load_c, df_load, ptw_IDs, pg_IDs, avg_rad_factor, approach, pg_rad_tol, tw_misori_angle, twin_tol, q_th, consider)

#%%**************************************To iterate over load-steps End********************************************************
    twin_pg_IDs = pd.DataFrame(twin_pg_IDs)
    print(f"# Twins at load-{load_c}:", len(twin_pg_IDs))
    toc = time.time()
    print(f"Time took in finding twins for load-step of {load_c}: {(toc-tic)/60:0.1f} min")
    #----------------------------This Part is just to estimate the scale/size factor -----------------------------------
    if approach == "wr":
        print(f"Avg norm size-factor:{np.average(scale_factor[f'Load-{load_c}']):0.4f} || Std: {np.std(scale_factor[f'Load-{load_c}']):0.4f}")
    else:
        print(f"Avg norm size-factor:{np.average(scale_factor[f'Load-{load_c}']):0.4f} || Std: {np.std(scale_factor[f'Load-{load_c}']):0.4f}")
    #---------------------------------------------------------------
    if len(twin_pg_IDs) ==0: #% If there are not any twins
        twin_pg_IDs = pd.DataFrame([[0,0]])
    #%% Saving twins load-wise as a dictionary
    tw_pg_dict[load_c] = [twin_pg_IDs.iloc[:,0].to_list(), twin_pg_IDs.iloc[:,1].to_list()]
    #%% Next load-step
    print(f"++"*25)
#------------------------------------------------------------------
#%% Saving the dictionary as ".csv" 
with open(f"{pth}/sample_{sam_n}/found_twins_loadwise/from_unmatched_grains_loadstep/pg_twin_IDs_load{load_s}_{load_f}_twangle{tw_misori_angle}_twtol{twin_tol}_radpg{pg_rad_tol}_{approach}_radf{avg_rad_factor}_{consider}.csv", "w") as twin_file: # wor = with out radius
    import csv
    writer  = csv.writer(twin_file)
    for key, value in tw_pg_dict.items():
        writer.writerow([key, value])

# ## to save scale-factor
# df_size_f = pd.DataFrame.from_dict(scale_factor, orient ="index").T
# df_size_f.to_csv(f"{pth}/sample_{sam_n}/found_twins_loadwise/from_unmatched_grains_icrmt_loadstep/pg_twin_IDs_load{load_s}_{load_f}_twintol{twin_tol}_radpg{pg_rad_tol}_{approach}_radf{avg_rad_factor}_sizef.csv", index = False)

# '''













