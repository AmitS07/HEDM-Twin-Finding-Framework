'''
Date: 3rd December 2025
Author: Amit, ACDM Lab, Emails: asingh38@crimosn.ua.edu, amitdravid@outlook.com
Description: 1. This script does calculate Resolved Shear Stress(RSS) analysis for parent-grains and its twins, for provided parnet-twin pair IDs. 
Note:1. For parent grain analysis, the parent-grain should be considered from load-step-0 i.e. initial load-steps, 
        If input load-range is not the from load-0, then one need to make sure, data should start from load-0 by adding it. 
        e.g. input load-range = [50,...300], then load_range_i = [0,25,50,...300], load_i = [0,25] this has to be done manually based on the increments of the loads
        and change it line here- "load_range_i = [load_i]+load_range" in this line
    2. Part-7 of this script plot the centroids (in real-space), and orientations(in orientation space) of the found and tracaked paret-twin pairs across all load steps.  
    2. The x-y limits of the plots has to be chekced and set accoridng to calculated RSS values. 
'''
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Part-0: Importing Libraries from python
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This is added to call the functions libraries/modules
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022') # Path to the function libraries
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from lib_twin_parent_matching import *
from lib_orientations_symmetries import *
from lib_functions_analysis import *
from lib_crystal_mechanics import *

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Part-1: Plotting requirments
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%Ploting requirment
fontsize = 13
plt.rcParams.update({"font.size":fontsize, "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,})
# plt.rcParams.update({"font.size":25})
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = [2.2,2.2]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-2: The Stiffness of the matrix
#%% Note: The Necessary and Sufficient condition for elastic Stability for HCP crystals involves following inequalities,
# C11 >|C12|, C44>0, C66>0 and 2C13^2 <C33(C11+C12)
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# C11, C12, C13, C33, C44 = 174.9, 87.10, 68, 189.4, 50.50 # Stiffness for Ti, Ref from a Book from Prof. Weaver's peronal library
# C11, C12, C13, C33, C44 = 169.7, 88.7, 61.7, 196.7, 42.5 # Stiffness for Ti64, all values are in GPa, Ref. Kasemer paper (https://doi.org/10.1016/j.actamat.2017.06.059)
C11, C12, C13, C33, C44 = 162.4, 92.0, 69.0, 180.7, 46.7 # Fisher's experimental https://journals.aps.org/pr/pdf/10.1103/PhysRev.135.A482
C_Ti = np.zeros((6,6)) # A 6*6 matrix
C_Ti[0,0] = C_Ti[1,1] = C11
C_Ti[0,1] = C_Ti[1,0] = C12
C_Ti[0,2] = C_Ti[2,0] = C_Ti[1,2] = C_Ti[2,1] = C13
C_Ti[2,2] = C33
C_Ti[3,3] = C_Ti[4,4] = C44
C_Ti[5,5] = (C11 - C12)/2

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-3: Schmid Tensor calculation for considered active slip and twin systems
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Pij_hcp_slip = hcp_slip_schmid_tensor('hcp',covera = 1.588) # Dr. Kasemer code to calculate pij: Schimd Tensor, covera = 1.588, as it was used duing HEDM (Hemant code)
Pij_hcp_twin = hcp_twin_schmid_tensor('hcp',covera = 1.588) # It is in the same format as for slip base don Dr. Kasemer code to calculate pij: Schimd Tensor

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-4: Importing variables and defining path and directories
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% 4.1: Variables
sam_n = eval(input("Enter sample number:"))
load_range = list(map(int, input(f"Enter all loads (space seprated) being analyzed: ").split()))
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")
twin_angle = eval(input("Enter Twin-angle(deg, depend on twin-type):"))#35
twin_tol = eval(input("Enter the twin-misori angle tol.(deg):")) # twin_tol = 2.5
pg_rad_tol = eval(input("Enter parent-grain radius tolerance(um):")) # pg_rad_tol = 10
avg_rad_factor = eval(input("Enter average-radius factor (1, if not any):")) #for sample3 = 2.19
approach = input("Enter type of approach('wr' if rad_pg>rad_tw, else 'wor'):") # approach = wor
consider = input(f"Enter type of analysis('all' or 'unique'):") # enter 'unique', if you want to consider no repated matched grans
compare = input("Enter PG-TW compare method(type: 'step_0' for 0th, else 'current'):") #step_0
#%%***************************************************************************************************
#%% 4.2: Path and directories
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
pth_tw_pg_dict = f"{pth}/sample_{sam_n}/found_twins_loadwise/from_unmatched_grains_loadstep/pg_twin_IDs_load{load_s}_{load_f}_twangle{twin_angle}_twtol{twin_tol}_radpg{pg_rad_tol}_{approach}_radf{avg_rad_factor}_{consider}.csv"
tw_pg_dict = read_csv_dictionary(pth_tw_pg_dict)
#4.3: Based on twin-IDs of "pg_twins-dict", findout corresponding "extra-matched grains" of unmatched grains for the load-range of k(th)-n(th)
#4.3.1: Reading the list of matched grains from list of "UNMATHCED" grains i.e. extra-grains
matchgrainIDs_um = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_fm_unmatched_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv",sep ="\t").dropna()
#4.3.2 Reading list of final-matched grains i.e. untimately matched grains across all load steps from kth-nth load step
final_matchgrainIDs = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_0_{load_f}.csv",sep ="\t").dropna()

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-5: Reading Parent-Twin pairs IDs
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% 5.0  making sure the parent-grain is considered from load step-0 i.e. load-0
#%% 5.1 Reading Parnet-Twin pairs for the load at which twin formed
twin_load = int(input("Enter load at which twin being analyze:"))
tw_ids_LS = {} #Step-wise twin-IDs for each load i.e. all twins, selected based on incremnt of load-steps 
pg_ids_LS = {} #Step-wise parent_grain-IDs for each load i.e. all parent-grains selected based on increment of load-steps
# load_c = load_s # load_c == current load-step
load_i = 0 ## at Step_0; load = 0, it cna be seted to any, but we caeful the given load-range
load_range_i = [load_i]+load_range #%% Added starting load which is "0" at load
load_select = [twin_load]# load_range; choose load at which or from which you want to analyse the twin
for load_c in load_select:
    ## Reading: Parent-twin pair data for k-nth load-range
    IDs_tw, IDs_pg = get_dictionary_values(tw_pg_dict, load_c)
    ## Saving common-grains based on twin-Ids for incremental load
    for i, (id_tw, id_pg) in enumerate(zip(IDs_tw, IDs_pg), start=1):
        ## Calculating step-wise twin-IDs, Twins we considered does not form at step_0, so we are not considring twin at step-0
        poten_tw_id = matchgrainIDs_um[matchgrainIDs_um[f"Load_{load_c}"]== id_tw] # poten_tw_id = potential twin iD
        tw_ids_LS[f"load_{load_c}_tw_{i}"] = df2list_specific_columns(load_range,poten_tw_id)#  "tw" = twinning based deformation, store IDs as a dict,
        ##---------------------------------------Parent----------------------------------------------------------------
        ## Calculating step-wise parent-IDs, since parent grains could be considered from "step_0" i.e. twin did not form at stpe_0. hence to compare what would has happend
        if compare == "step_0":
            pg_id = final_matchgrainIDs[final_matchgrainIDs[f"Load_{load_c}"]== id_pg] # parent grain
            pg_ids_LS[f"load_{load_c}_gr_{i}"] = df2list_specific_columns(load_range_i,pg_id)# "gr" = parent or neighbor grain, slip based deformation, saved as dictionary
        else: #need to check this
            pg_id = final_matchgrainIDs[final_matchgrainIDs[f"Load_{load_c}"]== id_pg] # parent grain
            pg_ids_LS[f"load_{load_c}_gr_{i}"] = df2list_specific_columns(load_range,pg_id)# "sl" = slip-base deformation, store IDs as a dict,
# print(pg_ids_LS)
# print(tw_ids_LS)
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-6: RSS calculation for parent and twin, by two approaches (i) wrt fix initial ori, (ii) Ori correspond to each grain
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Description: The RSS calculation could be done either (1) the parent-twin ids directly or (2) by sorting them based on load-step wise
#%% (1) Approach:
ls_n_tw = load_range.index(twin_load) # index of the load-step in load list at which twin form i.e. 
load_range_twin = load_range[ls_n_tw :]
idx_load_twins  = list(range(ls_n_tw, len(load_range))) # idx_load_twin = index of loads or load-step number from the step at which twin form to end of the load
#%% 6.1 The method by which the RSS will be calculated-
method = input(f"Enter the method of RSS calculation (type: 'fixori' else 'icrori'):")
pg_tw_pair_n = int(input(f"Enter #parent-twin pair that being analyzed:"))
#%%Note: In the following section, this parent-twin pair should be used to calculate the RSS
#%%**********************************************************************************************************************************
if method == "fixori":
    #%%///////////////////////////////////////////////////For Parent-grains///////////////////////////////////////////////////
    #%% Selecting load-steps
    load_i_pg = load_range_i[0] # Initial load = load_i_pg = 0; since parnet grain being analyzed from strating load
    ls_n_pg = load_range_i.index(load_i_pg) # load_step_number of parent from which 'slip' is being considered it is generally from load-step(ls) = 0 , i.e. starting or same as ls = ls_n_parent
    ##------------------------------------ Selecting initial/starting orientation of parent-grain---------------------------------------------
    pgID_i = pg_ids_LS[f"load_{twin_load}_gr_{pg_tw_pair_n}"][ls_n_pg]
    df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_i_pg}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
    ori_parent = list_to_matrix_nxn(df_final_matchgrains.loc[pgID_i]["O11":"O33"])
    print(f"Orientation matrix of Parent-ID {int(pgID_i)} at load {load_i_pg}-ls {ls_n_pg}:",ori_parent)
    cord_pg,strain_pg,stress_pg,u_pg, rss_pg_basal,rss_pg_prism,rss_pg_pyramI_a,rss_pg_pyramI_ca = StressStrainEnergyRSS_loadwise_fixori(pth,sam_n,load_range_i,twin_load,ls_n_pg,pg_tw_pair_n,ori_parent,pg_ids_LS,C_Ti,Pij_hcp_slip,"all_slip")
    ##==================================================Using CLASS===================================================
    # obj_slip = StressStrainEnergyRSS(pth, sam_n, C_Ti,Pij_hcp_slip)
    # cord_pg,strain_pg,stress_pg,rss_pg_basal,rss_pg_prism,rss_pg_pyramI_a,rss_pg_pyramI_ca = obj_slip.compute(load_range_i,ls_n_pg,pg_tw_pair_n,ori_parent,pgIDs_sorted_ls,"all_slip")
    ##================================================================================================================
    #%%///////////////////////////////////////////////////For Twins///////////////////////////////////////////////////
    load_i_tw = twin_load
    ##------------------------------------------Selecting orientation of a load-step at which twin form-----------------------------------
    twID_i = tw_ids_LS[f"load_{load_i_tw}_tw_{pg_tw_pair_n}"][ls_n_tw]
    df_finaltw_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_i_tw}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
    ori_twin = list_to_matrix_nxn(df_finaltw_matchgrains.loc[twID_i]["O11":"O33"])
    print(f"Orientation matrix of Twin-ID {twID_i} at load {load_i_tw}-ls {ls_n_tw}:",ori_twin)
    ###---------------------------------------------------------------------------------------------------------------------------------
    cord_tw,strain_tw,stress_tw,u_tw, rss_tw_t1, rss_tw_t2 = StressStrainEnergyRSS_loadwise_fixori(pth,sam_n,load_range_twin,twin_load,ls_n_tw,pg_tw_pair_n,ori_twin,tw_ids_LS,C_Ti,Pij_hcp_twin,"tensile_twins")
elif method == "icrori":
    #%%///////////////////////////////////////////////////For Parent-grains///////////////////////////////////////////////////
    #%% Selecting load-steps
    load_i_pg = load_range_i[0] # Initial load = load_i_pg = 0; since parnet grain being analyzed from strating load
    ls_n_pg = load_range_i.index(load_i_pg) # load_step_number of parent from which 'slip' is being considered it is generally from load-step(ls) = 0 , i.e. starting or same as ls = ls_n_parent
    ##------------------------------------ Selecting initial/starting orientation of parent-grain---------------------------------------------
    cord_pg,ori_pg,strain_pg,stress_pg,u_pg, rss_pg_basal,rss_pg_prism,rss_pg_pyramI_a,rss_pg_pyramI_ca = StressStrainEnergyRSS_loadwise(pth,sam_n,load_range_i,twin_load,ls_n_pg,pg_tw_pair_n,pg_ids_LS,C_Ti,Pij_hcp_slip,"all_slip")
    #%%///////////////////////////////////////////////////For Twins///////////////////////////////////////////////////
    load_i_tw = twin_load
    ##------------------------------------------Selecting orientation of a load-step at which twin form-----------------------------------
    cord_tw,ori_tw,strain_tw,stress_tw,u_tw, rss_tw_t1, rss_tw_t2 = StressStrainEnergyRSS_loadwise(pth,sam_n,load_range_twin,twin_load,ls_n_tw,pg_tw_pair_n,tw_ids_LS,C_Ti,Pij_hcp_twin,"tensile_twins")
''' 
#%% (2) approcach: the parent and twin-IDs after being sorted and all twin-ids index based are being estimated from the initial load-step load-i
# #%% 5.2 Sorting the parent grains-IDs
if compare == "step_0":
    pgIDs_sorted_ls = sorting_pgIDs_loadwise(load_range_i, pg_ids_LS)
else:
    pgIDs_sorted_ls = sorting_pgIDs_loadwise(load_range, pg_ids_LS)
# #%%***************************************************************************************************
# #%% 5.3 Sorting the Twin-IDs
ls_n_tw = load_range_i.index(twin_load) # index of the load-step in load list at which twin form i.e. 
load_range_twin = load_range_i[ls_n_tw :] # load_range_twin = list of loads from which twin got formed to the end of load-steps
idx_load_twins  = list(range(ls_n_tw, len(load_range_i))) # idx_load_twin = index of loads or load-step number from the step at which twin form to end of the load
twIDs_sorted_ls = sorting_twIDs_loadwise(idx_load_twins,load_range_twin,tw_ids_LS)

#%% 6.1 The method by which the RSS will be calculated-
method = input(f"Enter the method of RSS calculation (type: 'fixori' else 'ori'):")
# deformation_type = input("Enter the deformation type(type 'all_slip' or 'tensile_twins'):")
pg_tw_pair_n = int(input(f"Enter #parent-twin pair that being analyzed:"))

#%%******************************************
if method == "fixori":
    #%%///////////////////////////////////////////////////For Parent-grains///////////////////////////////////////////////////
    #%% Selecting load-steps
    load_i_pg = load_range_i[0] # Initial load = load_i_pg = 0; since parnet grain being analyzed from strating load
    ls_n_pg = load_range_i.index(load_i_pg) # load_step_number of parent from which 'slip' is being considered it is generally from load-step(ls) = 0 , i.e. starting or same as ls = ls_n_parent
    ##------------------------------------ Selecting initial/starting orientation of parent-grain---------------------------------------------
    pgID_i = pgIDs_sorted_ls[f"ls_{ls_n_pg}"][pg_tw_pair_n]
    df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_i_pg}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
    ori_parent = list_to_matrix_nxn(df_final_matchgrains.loc[pgID_i]["O11":"O33"])
    print(f"Orientation matrix of Parent-ID {int(pgID_i)} at load {load_i_pg}-ls {ls_n_pg}:",ori_parent)
    cord_pg,strain_pg,stress_pg,rss_pg_basal,rss_pg_prism,rss_pg_pyramI_a,rss_pg_pyramI_ca = stress_strain_energy_rss_loadstepwise_fixori(pth,sam_n,load_range_i,twin_load,ls_n_pg,pg_tw_pair_n,ori_parent,pgIDs_sorted_ls,C_Ti,Pij_hcp_slip,"all_slip")
    ##==================================================Using CLASS===================================================
    # obj_slip = StressStrainEnergyRSS(pth, sam_n, C_Ti,Pij_hcp_slip)
    # cord_pg,strain_pg,stress_pg,rss_pg_basal,rss_pg_prism,rss_pg_pyramI_a,rss_pg_pyramI_ca = obj_slip.compute(load_range_i,ls_n_pg,pg_tw_pair_n,ori_parent,pgIDs_sorted_ls,"all_slip")
    ##================================================================================================================
    #%%///////////////////////////////////////////////////For Twins///////////////////////////////////////////////////
    load_i_tw = twin_load
    ##------------------------------------------Selecting orientation of a load-step at which twin form-----------------------------------
    twID_i = twIDs_sorted_ls[f"ls_{ls_n_tw}"][pg_tw_pair_n]
    df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_i_tw}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
    ori_twin = list_to_matrix_nxn(df_final_matchgrains.loc[twID_i]["O11":"O33"])
    print(f"Orientation matrix of Twin-ID {twID_i} at load {load_i_tw}-ls {ls_n_tw}:",ori_twin)
    ###---------------------------------------------------------------------------------------------------------------------------------
    cord_tw,strain_tw,stress_tw,rss_tw_t1, rss_tw_t2 = stress_strain_energy_rss_loadstepwise_fixori(pth,sam_n,load_range_twin,twin_load,ls_n_tw,pg_tw_pair_n,ori_twin,twIDs_sorted_ls,C_Ti,Pij_hcp_twin,"tensile_twins")
     ##==================================================Using CLASS===================================================
    # obj_twin = StressStrainEnergyRSS(pth, sam_n, C_Ti,Pij_hcp_twin)
    # cord_tw,strain_tw,stress_tw,rss_tw_t1, rss_tw_t2 = obj_twin.compute(load_range_twin,ls_n_tw,pg_tw_pair_n,ori_twin,twIDs_sorted_ls,"tensile_twins")
    ##================================================================================================================
else: 
    print(f"Opps! RSS calculations with updated/current orientation has not been set-up yet for this approach")
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-7: Plotting the centroidal positions of parent and twins to veryfy the spatial distribution of grains
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%7.1  Defining colors and load-steps
colrs = ["tab:blue","tab:orange", "tab:green", "tab:purple", "tab:gray", "tab:cyan", "royalblue", "tab:red", "tab:olive", "tab:pink","tan", "indigo","#9A0EEA", "sienna", "black", "crimson", "lavender"]
linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (10, 5)), (0, (3, 1, 1, 1)), (0, (5, 5, 1, 5)), (0, (10, 3, 1, 3))]
## order of line stsyles::5. densely dotted,6. densely dashed, 7. dash-dot spaced,8. loosely dotted,9. loosely dashed,10. densely dashdot,11. dash-dot dash ,12. dash-long, dot-short
#%% Note:: For each slip-systems, each seprate plots will be geberated
load_steps = np.arange(0, len(load_range_i))
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ''' 
## Ploting grains's centroidal positions
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection = "3d")
ax.set_box_aspect((1, 1, 0.6)) # The dimension of the box selected based on the domain of scan area
## Parent grains centroidal
for ii in load_steps:
    x_pt = cord_pg[f"ls_{ii}"][0]
    y_pt = cord_pg[f"ls_{ii}"][1]
    z_pt = cord_pg[f"ls_{ii}"][2]
    ax.scatter(x_pt, y_pt, z_pt, color = colrs[ii], marker = "o", linewidth = 0.5,facecolors='none', alpha = 0.5)
## Twins centroidal
for jj in idx_load_twins:
    x_tw = cord_tw[f"ls_{jj}"][0]
    y_tw = cord_tw[f"ls_{jj}"][1]
    z_tw = cord_tw[f"ls_{jj}"][2]
    ax.scatter(x_tw, y_tw, z_tw, color = colrs[jj], marker = "d", linewidth = 0.5, facecolors='none', alpha = 0.5)
ax.set_xlabel(r'$x$ ($\mu m$)',labelpad=2) #labelpad=20
ax.set_ylabel(r'$y$ ($\mu m$)',labelpad=2)
ax.set_zlabel(r'$z$ ($\mu m$)', rotation=90,labelpad=2)
ticks = np.arange(-400,401,200)
ax.set_xlim([-500,500])
ax.set_ylim([-500,500])
ax.set_zlim([-400,400])
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.grid(True)
grid_style = {'color': (0, 0, 0, 0.1), 'linestyle': '-', 'linewidth': 0.5} ## color-code(initial,3-digit), and last is transpaency 0.3
ax.xaxis._axinfo['grid'].update((grid_style))
ax.yaxis._axinfo['grid'].update((grid_style))
ax.zaxis._axinfo['grid'].update((grid_style))
ax.margins(0) # To reduce the gap between the axis ticks and marks
ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)
# ax.view_init(elev=15, azim=315)
# pg_patch = mlines.Line2D([],[], marker = 'o', linestyle='None', fillstyle='none', color = "black", label = "PG") # [],[] these are just to represent empty "x" data and "y" data
# tw_patch = mlines.Line2D([],[],marker = 'd', linestyle='None',fillstyle='none',color = "black",label = f"Twin")
# ax.legend(handles = [tw_patch, pg_patch],frameon = False,loc='upper right',fontsize=15) # for: pg-and twin both
plt.savefig(f'{pth}/sample_{sam_n}/images/parent_twin_pair/coo_pg_tw_pair{pg_tw_pair_n}_twinned{twin_load}_misori{twin_angle}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()
## Ploting grains's orientations in Rodrigues space
if method == "icrori":
    fig_rf, ax_rf = plot_rodrigues_hcp_funda_region()
    for ii in load_steps:
        #% For parent grains
        quat_pg = rot2quat_scipy(ori_pg[f"ls_{ii}"])
        quat_pg_fr = Hexa_funda_region(quat_pg)
        rod_pg = quat2rod(quat_pg_fr)
        r1_pg, r2_pg, r3_pg = rod_pg[0], rod_pg[1], rod_pg[2]
        ax_rf.scatter(r1_pg,r2_pg,r3_pg, marker = "o", color = colrs[ii], facecolors='none', linewidth = 0.5, alpha = 0.5) #edgecolor='k'
    for jj in idx_load_twins:
        #% For found twin
        quat_tw = rot2quat_scipy(ori_tw[f"ls_{jj}"])
        quat_tw_fr = Hexa_funda_region(quat_tw)
        rod_tw = quat2rod(quat_tw_fr)
        r1_tw, r2_tw, r3_tw = rod_tw[0], rod_tw[1], rod_tw[2]
        ax_rf.scatter(r1_tw,r2_tw,r3_tw, marker = "d", color = colrs[jj], facecolors='none', linewidth = 0.5, alpha = 0.5) #facecolors='none', color = colrs[jj]
    plt.savefig(f'{pth}/sample_{sam_n}/images/parent_twin_pair/oris_pg_tw_pair{pg_tw_pair_n}_twinned{twin_load}_misori{twin_angle}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
    plt.close()
else:
    print(f"Warning!! for 'fixori' the orientations are fix, no need to show its evolution")
    print(20*"***")
# '''
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-8: Plotting the RSS results 
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% 8.1: Basal +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# fig = plt.figure()
# ax = fig.add_subplot()
fig = plt.figure(figsize=(2.5, 2.5))
ax =  fig.add_subplot()
ax.set_box_aspect(1)

xpos = np.arange(len(load_range_i[:-2]))# This is to make the x-axis equal sapcing, hence this work as x-values, and labels will be load_range_i
ss = f"basal"
basal_ss = [r"$(0001)[11\bar{2}0]$",r"$(0001)[\bar{1}2\bar{1}0]$",r"$(0001)[\bar{2}110]$"]
for bl in range(len(basal_ss)):
    trss = [rss_pg_basal[f"ls_{ls}"][0][bl] for ls in load_steps]
    ax.plot(xpos, trss[:-2], color = colrs[bl], marker = "h", ms = 5, mew=0, linestyle = linestyles[bl], linewidth = 1, alpha = 0.7, label = basal_ss[bl]) #x = load_range_i[:-2], to remobe last two didgit
## Plotting
# ax.legend(frameon = False, ncol = 1,fontsize='x-small')
y_ticks = np.arange(0,355,50)
ax.set_yticks(y_ticks)
ax.set_ylim([-5, 250])
# ax.set_xticks(load_range_i)
# ax.set_xticklabels(load_range_i,rotation=90, ha='center') #x = 'load_range_i' or e_pg
wanted_xticks = [0, 50, 100, 155, 185, 215, 245, 275]
want_xtick_labels = [str(v) if v in wanted_xticks else "" for v in load_range_i[:-2]]
ax.set_xticks(xpos)
ax.set_xticklabels(want_xtick_labels, rotation=90, ha='center')
# ax.set_xticklabels(load_range_i[:-2],rotation=90, ha='center') #x = 'load_range_i' or e_pg
# ax.set_xlim([-10,285])
ax.set_xlim(-0.5, len(xpos) - 0.5)
# plt.tight_layout()
ax.set_xlabel(r"Applied load $(N)$")
ax.set_ylabel(r"$\tau_{rss}$ $(MPa)$")
# plt.show()
plt.savefig(f'{pth}/sample_{sam_n}/images/rss/rss_vs_load_twinned{twin_load}_misori{twin_angle}_{ss}_pg{pg_tw_pair_n}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.cla() # To clear the figure axes
# plt.close()
# exit(0)
#%% 8.2: Prismatic +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ss = f"prism"
prism_ss = [r"$(1\bar{1}00)[11\bar{2}0]$",r"$(10\bar{1}0)[\bar{1}2\bar{1}0]$",r"$(01\bar{1}0)[\bar{2}110]$"]
for pm in range(len(prism_ss)):
    trss = [rss_pg_prism[f"ls_{ls}"][0][pm] for ls in load_steps]
    ax.plot(xpos, trss[:-2], color = colrs[pm], marker = "s", ms = 5,mew=0, linestyle = linestyles[pm], linewidth = 1, alpha = 0.7, label = prism_ss[pm])
    ## plt.plot(e_pg, trss, color = colrs[pm], marker = "s", ms = 10, linestyle = linestyles[pm], linewidth = 1.5, alpha = 0.7, label = prism_ss[pm])
## Plotting
# ax.legend(frameon = False, ncol = 1,fontsize='x-small')
ax.set_yticks(y_ticks)
ax.set_ylim([-5, 250])
# ax.set_xticks(load_range_i)
# ax.set_xticklabels(load_range_i,rotation=90, ha='center') #x = 'load_range_i' or e_pg
ax.set_xticks(xpos)
ax.set_xticklabels(want_xtick_labels, rotation=90, ha='center')
# ax.set_xticks(xpos)
# ax.set_xticklabels(load_range_i[:-2],rotation=90, ha='center') #x = 'load_range_i' or e_pg
# ax.set_xlim([-10,285])
ax.set_xlim(-0.5, len(xpos) - 0.5)
# plt.tight_layout()
ax.set_xlabel(r"Applied load $(N)$")
ax.set_ylabel(r"$\tau_{rss}$ $(MPa)$")
# plt.show()
plt.savefig(f'{pth}/sample_{sam_n}/images/rss/rss_vs_load_twinned{twin_load}_misori{twin_angle}_{ss}_pg{pg_tw_pair_n}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.cla()
#%% 8.3: Pyramidal (c+a) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ss = f"pyramI_ca"
pyramI_ca_ss = [r"$(1\bar{1}01)[\bar{1}2\bar{1}3]$",r"$(1\bar{1}01)[\bar{2}113]$",r"$(\bar{1}101)[1\bar{2}13]$", r"$(\bar{1}101)[2\bar{1}\bar{1}3]$",
            r"$(10\bar{1}1)[\bar{1}\bar{1}23]$", r"$(10\bar{1}1)[\bar{2}113]$", r"$(\bar{1}011)[11\bar{2}3]$", r"$(\bar{1}011)[2\bar{1}\bar{1}3]$", 
            r"$(01\bar{1}1)[\bar{1}\bar{1}23]$", r"$(01\bar{1}1)[1\bar{2}13]$",r"$(0\bar{1}11)[\bar{1}2\bar{1}3]$",r"$(0\bar{1}11)[11\bar{2}3]$"]
for pyIca in range(len(pyramI_ca_ss)):
    trss = [rss_pg_pyramI_ca[f"ls_{ls}"][0][pyIca] for ls in load_steps]
    ax.plot(xpos, trss[:-2],color = colrs[pyIca], marker = "d", ms = 5, mew=0, linestyle = linestyles[pyIca], linewidth = 1, alpha = 0.7,label = pyramI_ca_ss[pyIca])
    # plt.plot(e_pg, trss,color = colrs[pl], marker = "d", ms = 10, linestyle = linestyles[pl], linewidth = 1.5, alpha = 0.7,label = pyram_ss[pl])
## Plotting
# ax.legend(frameon = False, ncol = 2,fontsize='x-small')
ax.set_yticks(y_ticks)
ax.set_ylim([-5, 250])
# ax.set_xticks(load_range_i)
# ax.set_xticklabels(load_range_i,rotation=90, ha='center') #x = 'load_range_i' or e_pg
ax.set_xticks(xpos)
ax.set_xticklabels(want_xtick_labels, rotation=90, ha='center')
# ax.set_xticks(xpos)
# ax.set_xticklabels(load_range_i[:-2],rotation=90, ha='center') #x = 'load_range_i' or e_pg
# ax.set_xlim([-10,285])
ax.set_xlim(-0.5, len(xpos) - 0.5)
# plt.tight_layout()
ax.set_xlabel(r"Applied load $(N)$")
ax.set_ylabel(r"$\tau_{rss}$ $(MPa)$")
# plt.show()
plt.savefig(f'{pth}/sample_{sam_n}/images/rss/rss_vs_load_twinned{twin_load}_misori{twin_angle}_{ss}_pg{pg_tw_pair_n}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.cla()
plt.close()
#%% 8.4: T2 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ss = f"T2"
# t2_ts = [r"$(11\bar{2}1)[\bar{1}\bar{1}26]$",r"$(1\bar{2}11)[\bar{1}2\bar{1}6]$",r"$(\bar{2}111)[2\bar{1}\bar{1}6]$",
#          r"$(\bar{1}\bar{1}21)[11\bar{2}6]$",r"$(21\bar{1}1)[\bar{2}\bar{1}16]$",r"$(\bar{1}2\bar{1}1)[1\bar{2}16]$"]
# for tt2 in range(len(t2_ts)):
#     trss = [rss_tw_t2[f"ls_{ls}"][0][tt2] for ls in idx_load_twins]
#     plt.plot(load_range_twin[:-2], trss[:-2],color = colrs[tt2], marker = "d", ms = 5, linestyle = linestyles[tt2], linewidth = 1, alpha = 0.7,label = t2_ts[tt2])
# ## Plotting
# # ax.legend(frameon = False, ncol = 2,fontsize='x-small')
# ax.set_yticks(y_ticks)
# ax.set_ylim([-5, 250])
# ax.set_xticks(load_range_i)
# ax.set_xticklabels(load_range_i,rotation=90, ha='center') #x = 'load_range_i' or e_pg
# ax.set_xlim([-10,285])
# # plt.tight_layout()
# ax.set_xlabel(r"Applied load $(N)$")
# ax.set_ylabel(r"$\tau_{rss}$ $(MPa)$")
# plt.savefig(f'{pth}/sample_{sam_n}/images/rss/rss_vs_load_twinned{twin_load}_{ss}_pg{pg_tw_pair_n}_load{load_s}_{load_f}_fixori.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
# plt.close()
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-9: Plotting strain-energy density
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
''' 
fig = plt.figure()
ax =  fig.add_subplot()
ax.set_box_aspect(1)
## for parent grain
u_parent = [u_pg[f"ls_{ii}"]  for ii in load_steps]
## for twin
u_twin = [u_tw[f"ls_{jj}"]for jj in idx_load_twins]
## Ploting the Strain energies 
## for parent grain
ax.plot(load_range_i, u_parent, color = "b", marker = "o", linestyle = "--") # x = load_range_i or e_pg (strain_pg) equivalent strain 
# ax.plot(e_pg, u_parent, color = "b", marker = "o", linestyle = "--")
##-----------------------------------------------------------------------------------------
## for twin
ax.plot(load_range_twin , u_twin, color = "g", marker = "d", linestyle = "-.") # x = load_range_twin or e_tw (strain_tw)
# ax.plot(e_tw , u_twin, color = "g", marker = "d", linestyle = "-.")
##-----------------------------------------------------------------------------------------
##***** Set the axis for load-range
# ax.set_xticks(xpos)
# ax.set_xticklabels(want_xtick_labels, rotation=90, ha='center')
# ax.set_xlim(-0.5, len(xpos) - 0.5)
ax.set_xlabel(r"Applied load $(N)$")
plt.xticks(load_range_i, rotation=90, ha='center')
##+++++++++++++++++++++++++++
ax.set_ylabel(r'$U$ $(MPa)$', labelpad=10) # /U_{0}
ax.set_ylim([0,1200])
##% Putting-legends
# pg_patch = mlines.Line2D([],[], marker = 'o', linestyle='None', color = "black", label = "PG") # [],[] these are just to represent empty "x" data and "y" data
# tw_patch = mlines.Line2D([],[],marker = 'd', linestyle='None',color = "black",label = f"Twin")
# ax.legend(handles = [tw_patch, pg_patch],frameon = False,loc = "upper left",fontsize=15) # for: pg-and twin both
plt.savefig(f'{pth}/sample_{sam_n}/images/strain_energy/u_vs_load_pg_tw_pair{pg_tw_pair_n}_twinned{twin_load}_misori{twin_angle}_load{load_s}_{load_f}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()  
'''






