'''
Date: 21st Novermber 2026, (It is date of export from jupyter notbook, actually, it was written in Novemeber-2025)
Author: Amit, ACDM Lab, Emails: asingh38@crimosn.ua.edu, amitdravid@outlook.com
Description: 1. This script does calculate Resolved Shear Stress(RSS) analysis of neighbor-grains, for provided parnet-twin pair. 
Note:1. The neighbor grains corresponding to the parent grain selected from load-step-0 i.e. initial load-steps, 
        If input load-range is not the from load-0, then one need to make sure, data should start from load-0 by adding it. 
        e.g. input load-range = [50,...300], then load_range_i = [0,25,50,...300], load_i = [0,25] this has to be done manually based on the increments of the loads
        and change it line here- "load_range_i = [load_i]+load_range" in this line
    2. Part-7 of this script plot the centroids (in real-space), and orientations(in orientation space) of the found and tracaked paret-twin pairs across all load steps.  
    3. The x-y limits of the plots has to be chekced and set accoridng to calculated RSS values. 
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
#%%***************************************************************************************************
#4.3: Based on twin-IDs of "pg_twins-dict", findout corresponding "extra-matched grains" of unmatched grains for the load-range of k(th)-n(th)
#4.3.1: Reading the list of matched grains from list of "UNMATHCED" grains i.e. extra-grains
matchgrainIDs_um = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_fm_unmatched_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv",sep ="\t").dropna()
#4.3.2 Reading list of final-matched grains i.e. untimately matched grains across all load steps from kth-nth load step
final_matchgrainIDs = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_0_{load_f}.csv",sep ="\t").dropna()
#%%***************************************************************************************************
#%% 4.4: Loading parent and neighbor grains IDs data after finding fron Neper
twin_load = int(input("Enter the load at which twin form:"))
wt_method = input(f"Enter weight-method for grainsize('rad'):") #rad
pg_tw_pair_n= int(input("Enter parent-twin pair#:")) #Parent-twin_pair_number, begin with 1, it is possible that at per load we may have found more than one twin, Here it is called just to know which pair we are working
parent_id = np.loadtxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/parent/pg_pair{pg_tw_pair_n}_ids_misori{twin_angle}_twload{twin_load}.txt")
neighbors_ids = np.loadtxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/neighbors/nghbr_ids_pg_pair{pg_tw_pair_n}_misori{twin_angle}_twload{twin_load}_w{wt_method}.txt")
print("+++"*25)
print(f'Total number of neighbors: {len(neighbors_ids)}')
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-5: Which neighbor? details of that particular neighbor and its trcing from load-0
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%5.1: Tracing neighbor grains from starting load
load_i = 0 ## load_i = initail load; load = 0, it can be set to any, but we caeful the given load-range
load_range_i = [load_i]+load_range #%% Added starting load which is "0" at load
##Note:: These neighbor-grains are corresponding to a parent grain which got twin a load-X.But RSS analysis must be carried out from the load-0
## to check or track neigjbor grains RSS from the begning. So, these neighbor-ids found at load-x must be mapped from load-0 
ng_ids_ls = {}
for i, id_ng in enumerate(neighbors_ids, start=1):
    ## Calculating step-wise parent-IDs, since parent grains could be considered from "step_0" i.e. twin did not form at stpe_0. hence to compare what would has happend
    if compare == "step_0":
        ngID = final_matchgrainIDs[final_matchgrainIDs[f"Load_{twin_load}"]== id_ng] # neighbor grain
        ng_ids_ls[f"load_{twin_load}_gr_{i}"] = df2list_specific_columns(load_range_i,ngID)#  Neighbor grain(gr) id as trcaked from load-step 0
    else:
        print(f"RSS analysis must be carried out from the starting load!!")
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-6: RSS analysis and plots of selected neighbor-grain corresponding to the parent grain at which twin might have formed
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
method = input(f"Enter the method of RSS calculation (type: 'fixori' else 'icrori'):")
#ng_n = int(input("Enter nieghbor number(which neighbor):")) #Which neighbor? neighbor,  
#%Note: The following section is to loop over all the neighbor grains, if one want to do it for one-by-one analysis of RSS, then loop has to be remove and above line has to be uncommented
##6.1: Looping over for all the neghbor grains
for ng_n in range(1,len(neighbors_ids)+1):
    print(f"Neighbor-grain#{ng_n}||ID {neighbors_ids[ng_n-1]}")
    if method == "fixori":
        #%% Selecting starting load step
        ls_i_ng = load_range_i.index(load_i) # load_step_number of neighbor from which 'slip' is being considered it is generally from load-step(ls) = 0 , load_i_idx_ng(index of inital load) = 0
        ##------------------------------------ Selecting initial/starting orientation of neighbor-grain---------------------------------------------
        ngID_i = ng_ids_ls[f"load_{twin_load}_gr_{ng_n}"][ls_i_ng] # neighbor grain at starting or initial condition 
        df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_i}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
        ori_nghbr = list_to_matrix_nxn(df_final_matchgrains.loc[ngID_i]["O11":"O33"])
        print(f"Orientation of Neighbor#{ng_n}-GrainID#{int(ngID_i)} at load {load_i}:",ori_nghbr)
        ##------------------------------------ Looping over all the load-steps---------------------------------------------
        cord_ng,strain_ng,stress_ng,u_ng,rss_ng_basal,rss_ng_prism,rss_ng_pyramI_a,rss_ng_pyramI_ca = StressStrainEnergyRSS_loadwise_fixori(pth,sam_n,load_range_i,twin_load,ls_i_ng,ng_n,ori_nghbr,ng_ids_ls,C_Ti,Pij_hcp_slip,"all_slip")
    elif method == "icrori":
        #%% Selecting starting load step
        ls_i_ng = load_range_i.index(load_i) # load_step_number of neighbor from which 'slip' is being considered it is generally from load-step(ls) = 0 , load_i_idx_ng(index of inital load) = 0
        cord_ng,ori_ng,strain_ng,stress_ng,u_ng, rss_ng_basal,rss_ng_prism,rss_ng_pyramI_a,rss_ng_pyramI_ca = StressStrainEnergyRSS_loadwise(pth,sam_n,load_range_i,twin_load,ls_i_ng,ng_n,ng_ids_ls,C_Ti,Pij_hcp_slip,"all_slip")
    #%%6.2: Plotting the RSS results 
    #%%7.1  Defining colors and load-steps
    colrs = ["tab:blue","tab:orange", "tab:green", "tab:purple", "tab:gray", "tab:cyan", "royalblue", "tab:red", "tab:olive", "tab:pink","tan", "indigo","#9A0EEA", "sienna", "black", "crimson", "lavender"]
    linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (10, 5)), (0, (3, 1, 1, 1)), (0, (5, 5, 1, 5)), (0, (10, 3, 1, 3))]
    ## order of line stsyles::5. densely dotted,6. densely dashed, 7. dash-dot spaced,8. loosely dotted,9. loosely dashed,10. densely dashdot,11. dash-dot dash ,12. dash-long, dot-short
    #%% Note:: For each slip-systems, each seprate plots will be geberated
    load_steps = np.arange(0, len(load_range_i))
    #%%7.2: RSS plots
    fig = plt.figure(figsize=(2.5, 2.5))
    ax =  fig.add_subplot()
    ax.set_box_aspect(1)
    #%% 7.2.1: Basal +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    xpos = np.arange(len(load_range_i[:-2]))# This is to make the x-axis equal sapcing, hence this work as x-values, and labels will be load_range_i
    ss = f"basal"
    basal_ss = [r"$(0001)[11\bar{2}0]$",r"$(0001)[\bar{1}2\bar{1}0]$",r"$(0001)[\bar{2}110]$"]
    for bl in range(len(basal_ss)):
        trss = [rss_ng_basal[f"ls_{ls}"][0][bl] for ls in load_steps]
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
    plt.savefig(f'{pth}/sample_{sam_n}/images/rss/neighbor/rss_vs_load_nghbr{ng_n}_twin{twin_load}_misori{twin_angle}_{ss}_pgpair{pg_tw_pair_n}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
    plt.cla() # To clear the figure axes
    # plt.close()
    # exit(0)
    #%% 8.2: Prismatic +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ss = f"prism"
    prism_ss = [r"$(1\bar{1}00)[11\bar{2}0]$",r"$(10\bar{1}0)[\bar{1}2\bar{1}0]$",r"$(01\bar{1}0)[\bar{2}110]$"]
    for pm in range(len(prism_ss)):
        trss = [rss_ng_prism[f"ls_{ls}"][0][pm] for ls in load_steps]
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
    plt.savefig(f'{pth}/sample_{sam_n}/images/rss/neighbor/rss_vs_load_nghbr{ng_n}_twin{twin_load}_misori{twin_angle}_{ss}_pgpair{pg_tw_pair_n}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
    plt.cla()
    #%% 8.3: Pyramidal (c+a) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ss = f"pyramI_ca"
    pyramI_ca_ss = [r"$(1\bar{1}01)[\bar{1}2\bar{1}3]$",r"$(1\bar{1}01)[\bar{2}113]$",r"$(\bar{1}101)[1\bar{2}13]$", r"$(\bar{1}101)[2\bar{1}\bar{1}3]$",
                r"$(10\bar{1}1)[\bar{1}\bar{1}23]$", r"$(10\bar{1}1)[\bar{2}113]$", r"$(\bar{1}011)[11\bar{2}3]$", r"$(\bar{1}011)[2\bar{1}\bar{1}3]$", 
                r"$(01\bar{1}1)[\bar{1}\bar{1}23]$", r"$(01\bar{1}1)[1\bar{2}13]$",r"$(0\bar{1}11)[\bar{1}2\bar{1}3]$",r"$(0\bar{1}11)[11\bar{2}3]$"]
    for pyIca in range(len(pyramI_ca_ss)):
        trss = [rss_ng_pyramI_ca[f"ls_{ls}"][0][pyIca] for ls in load_steps]
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
    plt.savefig(f'{pth}/sample_{sam_n}/images/rss/neighbor/rss_vs_load_nghbr{ng_n}_twin{twin_load}_misori{twin_angle}_{ss}_pgpair{pg_tw_pair_n}_load{load_s}_{load_f}_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
    plt.cla()
    plt.close()
    print("+++"*25)