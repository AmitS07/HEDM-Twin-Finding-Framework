'''
Date: 12th August 2024
Author: Amit, ACDM Lab
Description: This code will trace the 'found' twin/s across load steps based on \\
             given selected twin criteria for a given sample. 
             Input: The dataframe or data of found tiwns
             Output: It print the iteretively matched twins with its parent-grains
Note:1. The radius of parent-grains and twins are multiplied by "avg_rad_factor" i.e either 1( as raw data) or 
         what ever it comes after calculation, from volume of measured grains to volume of scanned layer)
    2. The parent grains were selected from the final-matched grains from the 0-300 load steps..
        not from the currrent starting load step that we are giving. 
'''
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++ Part:0 Importing required libaries and functions ++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This is added to call the functions libraries
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022') # Path to the function libraries
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from lib_twin_parent_matching import *
from lib_orientations_symmetries import *

#%%************************** Ploting requirment ********************
plt.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = [9, 9] #default: [6.4, 4.8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
# plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++ Part:1 Loading dataframes for matched parent-grains and potential twins +++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#1.1: Defining sample and related parameters
sam_n = eval(input("Enter sample number:"))
load_range = list(map(int, input(f"Enter all loads (space seprated) being analyzed: ").split()))
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")
twin_angle = eval(input("Enter Twin-angle(deg, depend on twin-type):"))#
twin_tol = eval(input("Enter the twin-misori angle tol.(deg):")) # twin_tol = 2.5
pg_rad_tol = eval(input("Enter parent-grain radius tolerance(um):")) # pg_rad_tol = 10
avg_rad_factor = eval(input("Enter average-radius factor (1, if not any):"))
approach = input("Enter type of approach('wr' if rad_pg>rad_tw, else 'wor'):")
consider = input(f"Enter type of analysis('all' or 'unique'):") # enter 'unique', if you want to consider no repated matched grans

#1.2: Defining path and directory
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
pth_tw_pg_dict = f"{pth}/sample_{sam_n}/found_twins_loadwise/from_unmatched_grains_loadstep/pg_twin_IDs_load{load_s}_{load_f}_twangle{twin_angle}_twtol{twin_tol}_radpg{pg_rad_tol}_{approach}_radf{avg_rad_factor}_{consider}.csv"
tw_pg_dict = read_csv_dictionary(pth_tw_pg_dict)

#1.3: Based on twin-IDs of "pg_twins-dict", findout corresponding "extra-matched grains" of unmatched grains for the load-range of k(th)-n(th)
#1.3.1: Reading the list of matched grains from list of "UNMATHCED" grains i.e. extra-grains
matchgrainIDs_um = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_fm_unmatched_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv",sep ="\t").dropna()
#1.3.2 Reading list of final-matched grains i.e. untimately matched grains across all load steps from kth-nth load step
final_matchgrainIDs = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_0_{load_f}.csv",sep ="\t").dropna()

colrs = ["tab:blue","tab:orange", "tab:green", "tab:purple", "tab:gray", "tab:cyan", "royalblue", "tab:red", "tab:olive", "tab:pink","tan", "indigo","#9A0EEA", "sienna", "black", "crimson", "lavender"]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++ Part-2: Checking found parent and twin pairs visually for a given load via plotting
# its centroidal in real-space and orientations in rodrigues space
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
'''
load_c = eval(input("Enter load of interest for parent-twin pair:"))# load_c = load at which parent-twin being selected
IDs_tw, IDs_pg = get_dictionary_values(tw_pg_dict, load_c) # Note: The dictionary contain 'twin_ids" on LHS  and "parent_ids" on RHS
#%%2.1: Estimating the centroid and orientation of parent grains
df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
coo_pg, ori_pg = {}, {}
for ii, id_n in enumerate(IDs_pg):
    coo_pg[f"ls_{ii}"] = df_final_matchgrains.loc[id_n]["X":"Z"]
    ori_pg[f"ls_{ii}"] = rot2mat(df_final_matchgrains.loc[id_n]["O11":"O33"]) ## This rotation matrix is in Crystal to Sample

coo_tw, ori_tw = {}, {}
for ii, id_n in enumerate(IDs_tw):
    coo_tw[f"ls_{ii}"] = df_final_matchgrains.loc[id_n]["X":"Z"]
    ori_tw[f"ls_{ii}"] = rot2mat(df_final_matchgrains.loc[id_n]["O11":"O33"]) ## This rotation matrix is in Crystal to Sample

#%%2.2: Ploting centroidal in real-space
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.set_box_aspect((1, 1, 0.6)) # The dimension of the box selected based on the domain of scan area
ax.grid(True)
grid_style = {'color': (0, 0, 0, 0.1), 'linestyle': '-', 'linewidth': 0.5} ## color-code(initial,3-digit), and last is transpaency 0.3
ax.xaxis._axinfo['grid'].update((grid_style))
ax.yaxis._axinfo['grid'].update((grid_style))
ax.zaxis._axinfo['grid'].update((grid_style))
for ii in range(len(IDs_pg)): # Note: len(IDs_pg) = len(IDs_tw)
    # For parent
    x_pg, y_pg, z_pg = coo_pg[f"ls_{ii}"][0], coo_pg[f"ls_{ii}"][1], coo_pg[f"ls_{ii}"][2]
    ax.scatter(x_pg, y_pg, z_pg, s = 50, marker = "o", color = colrs[ii], edgecolor='k', linewidth = 0.1, alpha = 0.7) #color = colrs[ii]
    # For twin
    x_tw, y_tw, z_tw = coo_tw[f"ls_{ii}"][0], coo_tw[f"ls_{ii}"][1], coo_tw[f"ls_{ii}"][2]
    ax.scatter(x_tw, y_tw, z_tw, s = 50, marker = "d", color = colrs[ii], edgecolor='k', linewidth = 0.1, alpha = 0.7) #color = colrs[jj],
#%% Plotting axis and labels details
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
# plt.tight_layout()
ax.margins(0) # To reduce the gap between the axis ticks and marks
ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)
plt.savefig(f'{pth}/sample_{sam_n}/images/parent_twin_pair/centroidal_all_pg_tw_pair_misoriangle{twin_angle}_twinned{load_c}_load{load_s}_{load_f}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()

#%%2.3: Ploting orientations Rodrigues space
fig_rf, ax_rf = plot_rodrigues_hcp_funda_region()
for ii in range(len(IDs_pg)): # Note: len(IDs_pg) = len(IDs_tw)
    #% For parent grains
    quat_pg = rot2quat_scipy(ori_pg[f"ls_{ii}"])
    quat_pg_fr = Hexa_funda_region(quat_pg)
    rod_pg = quat2rod(quat_pg_fr)
    r1_pg, r2_pg, r3_pg = rod_pg[0], rod_pg[1], rod_pg[2]
    ax_rf.scatter(r1_pg,r2_pg,r3_pg, s = 50, marker = "o", color = colrs[ii], edgecolor='k', linewidth = 0.1, alpha = 0.7)
    #% For found twin
    quat_tw = rot2quat_scipy(ori_tw[f"ls_{ii}"])
    quat_tw_fr = Hexa_funda_region(quat_tw)
    rod_tw = quat2rod(quat_tw_fr)
    r1_tw, r2_tw, r3_tw = rod_tw[0], rod_tw[1], rod_tw[2]
    ax_rf.scatter(r1_tw,r2_tw,r3_tw,s = 50, marker = "d", color = colrs[ii], edgecolor='k', linewidth = 0.1, alpha = 0.7) #facecolors='none', color = colrs[jj]
plt.savefig(f'{pth}/sample_{sam_n}/images/parent_twin_pair/oris_all_pg_tw_pair_misoriangle{twin_angle}_twinned{load_c}_load{load_s}_{load_f}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()
exit(0)
''' 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++ Part:3 "Tracing parent/twin" load/step-wise, listing twins and parent-grains for the a given starting load-step to final load-step+++++#
#Note: This part is very crucial, it does track the found parent-twin pairs from its from starting (or given load) to end of the load-step
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
tw_ids_LS = {} #Step-wise twin-IDs for each load i.e. all twins, selected based on incremnt of load-steps 
pg_ids_LS = {} #Step-wise parent_grain-IDs for each load i.e. all parent-grains selected based on increment of load-steps
load_c = load_s # load_c == current load-step
for load_c in load_range:
    ## Reading: Parent-twin pair data for k-nth load-range
    IDs_tw, IDs_pg = get_dictionary_values(tw_pg_dict, load_c) # Note: The dictionary contain 'twin_ids" on left hand side  and "parent_ids" on right hand side
    ## Saving common-grains based on twin-Ids for incremental load
    for i, (id_tw, id_pg) in enumerate(zip(IDs_tw, IDs_pg), start=1):
        poten_tw_id = matchgrainIDs_um[matchgrainIDs_um[f"Load_{load_c}"]== id_tw] # poten_tw_id = potential twin iD
        tw_ids_LS[f"LS_{load_c}_Twin_{i}"] = df2list_specific_columns(load_range,poten_tw_id)#  as a dict
        pg_id = final_matchgrainIDs[final_matchgrainIDs[f"Load_{load_c}"]== id_pg] # parent grain
        pg_ids_LS[f"LS_{load_c}_PG_{i}"] = df2list_specific_columns(load_range,pg_id)#  as a dict
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++ Part:4 Checking for "ultimately" same twin-across the load-steps: either with incremnt of load-step or relative to 1st (or given) load step +++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
''' 
#%%4.1 Looking for "same" twins with increment of load/ or with starting load step
load_c = load_s# load_c :: Current load_step
n_load_steps = len(load_range) # Total number of load-steps
##======== To set-up the matrix for twin-presence at each load-steps compare to starting load-step  ================
IDs_tw_s, IDs_pg_s = get_dictionary_values(tw_pg_dict, load_s) # This is to get the number of twins at given particular load-steps 
n_twins_at_load = int(len(IDs_tw_s))
size_twin_matrix = (n_twins_at_load,n_load_steps)
twin_available_matrix =  np.zeros(size_twin_matrix, dtype=int) 
#---------------------------------------------------------------------------------------------------------------------
method = input("Enter type of twin comparison ('start' or 'icrmtly':")
counter = 1
for load_c in load_range:
    ## Reading: Parent-twin pair data for 25-300 load-range
    if method == 'icrmtly':
        load_p = load_c[counter] # Previous_load_step = load_p; This could be used while comparing "incremntally" with load-steps
    else:
        load_p = load_s # = load_s, this could be used when need to comapre from starting load step
    IDs_tw_p, IDs_pg_p = get_dictionary_values(tw_pg_dict, load_p) # This is to read the IDs of previous load step "p" i.e. (k-1)th
    IDs_tw_c, IDs_pg_c = get_dictionary_values(tw_pg_dict, load_c) # This is to read the IDs of current load step "c" i.e. kth
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    out_twins = [] # To save the number of "OUT" twins
    in_twins = [] # To save the number of "IN" twins
    if load_c >load_s: #load_s = 25
        for j in range(1,len(IDs_tw_p)+1): # Looping over number of twins in "load_p"
            same_twin = list(set(tw_ids_LS[f"LS_{load_p}_Twin_{j}"]).intersection(IDs_tw_c)) # serching for same twins
            #---------------------------------
            # This is just to save "OUT" and "IN" twins
            if len(same_twin)==0: 
                out_twins.append(same_twin)
            else:
                in_twins.append(same_twin)
                twin_available_matrix[j-1,counter-1] = 1 # Just to add the 
            #---------------------------------
            print(f"Twins-IN at LS-{load_c} from LS_{load_p}_Twin_{IDs_tw_p[j-1]}:",same_twin)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print(20*"--")
    print(f"# In twins: {len(in_twins)}")
    print(f"# Out twins: {len(out_twins)}")
    print(24*"+++")
    counter +=1
##============================ To plot the "Twin-Binary matrix" =============================
twin_available_matrix[:,0] = 1
print(f"The Twins availability as binary matrix: \n {twin_available_matrix}")
cmap_reversed = plt.colormaps['binary'] # Thisnis to change the oder of the color-bar, reversed()
plt.matshow(twin_available_matrix, cmap=cmap_reversed)
#----------------------------------------------------------------
##% To find out x_ticks based on the load-steps indices of elements after first occurrence of load_s
load_list = load_range
xticks_id = [] 
for idx, load in enumerate(load_list):
    if load == load_s:
        xticks_id  = list(range(idx+1, len(load_list)+1))  # Get all indices after current position
        break  # Stop after first match
plt.xticks(range(n_load_steps), xticks_id)
#-----------------------------------------------------------------
# plt.xticks(range(n_load_steps), range(1, n_load_steps+1))
twin_ids = [int(id) for id in IDs_tw_s.tolist()]
y_labels = [f"Twin-[{id}]" for id in twin_ids] # This is to print the y_tick label 
plt.yticks(range(len(twin_ids)), y_labels)
plt.savefig(f'{pth}/sample_{sam_n}/images/twins_track/twin_binary_load{load_s}_{load_f}_twintol{twin_tol}_radpg{pg_rad_tol}_{approach}_radf{avg_rad_factor}_compare_{method}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()  
'''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++ Part:5 Cause for twin's dropout and re-appear on comparison to previous load step, (i) Spatial and/or (ii) misori ++#
#%% Note: Run Part-3, to run this section
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
''' 
IDs_tw_p, IDs_pg_p = get_dictionary_values(tw_pg_dict, load_s)
sf = []
load_c = load_s
counter = 0
for load_c in load_range:
    print(f"++++++++++++++++++++++++ Load_step: {load_c}++++++++++++++++++++++++")
    df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv",skiprows=1, sep = "\t")
    df_modif = df_modif.set_index(["%new_GrainID"])
    size_factor = []
    r_tw, r_pg = [], []
    for j in range(1, len(IDs_tw_p) + 1): # Interatve over for each twin per load steps i.e. number of twins for a load steps
#     for j,id_pg in zip(range(1,len(IDs_tw_p)+1),IDs_pg_p): # To loop over each twin
        same_twin = tw_ids_LS[f"LS_{load_s}_Twin_{j}"][counter] # twin for load
        same_pg = pg_ids_LS[f"LS_{load_s}_PG_{j}"][counter] # twin for load
        ## Reading twin's details
        rot_tw = df_modif.loc[same_twin]["O11":"O33"]
        quat_tw = rot2quat_scipy(rot2mat(rot_tw.to_list()))
        cord_tw = df_modif.loc[same_twin]["X":"Z"]
        rad_tw = df_modif.loc[same_twin]["GrainRadius"]*avg_rad_factor 
        r_tw.append(rad_tw)
        ## Reading parent-grain's details
        rot_pg = df_modif.loc[same_pg]["O11":"O33"]
        quat_pg = rot2quat_scipy(rot2mat(rot_pg.to_list()))
        cord_pg = df_modif.loc[same_pg]["X":"Z"]
        rad_pg = df_modif.loc[same_pg]["GrainRadius"]*avg_rad_factor
        r_pg.append(rad_pg)
        ## Criteria-analysis
        misori_tw = mis_ori(quat_tw, quat_pg)
        euclid_tw = euclidean_dist(cord_tw,cord_pg) #euclid_tw = np.linalg.norm(np.array(cord_tw)-np.array(cord_pg))
        poistion_tol = rad_pg+pg_rad_tol #Note: "avg_rad_factor" is already accounted 
        scale_factor = rad_pg/rad_tw # This is to check, whether rad_pg>rad_tw, there no need to multiply the "avg_rad_factor"
        size_factor.append(scale_factor) 
        sf.append(scale_factor)
        print(f"Twin:{same_twin}-PG:{same_pg} || misori:{'%.4f'% misori_tw}, euclid:{'%0.4f'%euclid_tw} // rad_tol:{'%0.4f'%poistion_tol}, size_factor:{'%0.4f'%scale_factor}")
    print(f"Scale-factor (rad_pg*m>rad_tw), m ={'%0.4f'% np.average(size_factor)}±{'%0.4f'% np.std(size_factor)}")
    counter += 1
print(r_pg)
print(r_tw)
print("-----")
print(size_factor)
print(len(sf),'%0.4f'% np.average(sf), '%0.4f'% np.std(sf))
'''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++ Part:6 For 2D and 3D visualization of selected twin with its parent grain ++++
# Note: Before, running this Part-6, you need to run Part-3
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
''' 
## Value scalculation for 2D-3D plot
tw_n = eval(input(f"Enter the twin-number(that you want to plot):"))
load_c = load_s 
n_load_steps = len(load_range) # Total number of load-steps
load_steps = np.arange(1, n_load_steps+1).tolist()
# Selecting parent-grain and twin pair for a given load, i.e. for starting-load
same_twin = tw_ids_LS[f"LS_{load_c}_Twin_{tw_n}"] # twin load-wise
same_pg = pg_ids_LS[f"LS_{load_c}_PG_{tw_n}"] # pg load-wise
# Defining-empty lists of variables
x_pg,y_pg,z_pg = [],[],[]
x_tw,y_tw,z_tw = [],[],[]
rad_pg = []
rad_tw = []
counter = 0
for load_c in load_range:
    # Reading the modified data for each load step
    df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv",skiprows=1, sep = "\t")
    df_modif = df_modif.set_index(["%new_GrainID"])
    ## For parent grain
    cord_pg = df_modif.loc[same_pg[counter]]["X":"Z"] # cord: cordinate of x,y,z
    x_pg.append(cord_pg[0]) # x -position
    y_pg.append(cord_pg[1]) # y -position
    z_pg.append(cord_pg[2]) # z -position
    rad_pg.append(df_modif.loc[same_pg[counter]]["GrainRadius"]*avg_rad_factor)
    ## For twin
    cord_tw = df_modif.loc[same_twin[counter]]["X":"Z"]
    x_tw.append(cord_tw[0])
    y_tw.append(cord_tw[1])
    z_tw.append(cord_tw[2])
    rad_tw.append(df_modif.loc[same_twin[counter]]["GrainRadius"]*avg_rad_factor)
    # print(f"Twin-{same_twin[counter]},PG-{same_pg[counter]} pair")
    counter += 1
## Toleranec: in spatial criteria
pg_rad_tol_err = [rad+pg_rad_tol for rad in rad_pg]
''' 

#%%6.2 Plotting the spatial distribution wrt load-steps
#%%-----------------------------------------------------------------
#%% 2D- plots for x, y & z positions, stacked together
'''
import matplotlib.lines as mlines
fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, sharey=False) ## Seting up the plot grid
ax1.scatter(load_steps, x_tw,color = "g", marker = "^" ) #label = f"Twin-{tw_n}: x"
ax1.scatter(load_steps, x_pg, color = "b", marker = "o")
ax1.errorbar(load_steps, x_pg,yerr= pg_rad_tol_err, fmt='none', ecolor = 'grey', capsize=5)
ax2.scatter(load_steps, y_tw, color = "g", marker = "^")
ax2.scatter(load_steps, y_pg, color = "b", marker = "o")
ax2.errorbar(load_steps, y_pg,yerr= pg_rad_tol_err, fmt='none', ecolor = 'grey', capsize=5)
ax3.scatter(load_steps, z_tw, color = "g", marker = "^") #label = "Twin", color = colors[i]
ax3.scatter(load_steps, z_pg, color = "b", marker = "o")  #label = "PG"
ax3.errorbar(load_steps, z_pg,yerr= pg_rad_tol_err, fmt='none', ecolor = 'grey', capsize=5) #label = r'$rad_{PG}+{tol}$'
# seleting limits and labels of 2D plots
# Set x-axis
ax3.set_xticks(load_steps)
ax3.set_xticklabels(load_steps)
ax3.set_xlabel(r'Load-Steps', labelpad=10)
# Set y-axis
ax1.set_ylim([min(x_tw)-100,max(x_tw)+100])
ax2.set_ylim([min(y_tw)-100,max(y_tw)+100])
ax3.set_ylim([min(z_tw)-100,max(z_tw)+100])
ax1.set_ylabel(r'x-position ($\mu m$)', labelpad=10)
ax2.set_ylabel(r'y-position ($\mu m$)', labelpad=10)
ax3.set_ylabel(r'z-position ($\mu m$)', labelpad=10)
# Putting-legends
pg_patch = mlines.Line2D([],[], marker = 'o', linestyle='None', color = "black", label = "PG") # [],[] these are just to represent empty "x" data and "y" data
tw_patch = mlines.Line2D([],[],marker = '^', linestyle='None',color = "black",label = f"Twin-{tw_n}")
ax3.legend(handles = [tw_patch, pg_patch],frameon = False,loc='upper right',fontsize=15) # for: pg-and twin both
plt.savefig(f'{pth}/sample_{sam_n}/images/twins_track/trace_pg_twin{tw_n}_pair_load{load_s}_{load_f}_misoritol{twin_tol}_radpgtol{pg_rad_tol}_avgradf{avg_rad_factor}_{approach}_2D.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()  
'''
#%%-----------------------------------------------------------------
#%% 3D- plots for x, y & z positions, stacked together
'''
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(x_pg,y_pg,z_pg, c =load_steps, cmap = 'viridis', marker = "o", s = rad_pg, alpha = 0.5) # color =colors[i]
ax.scatter(x_tw,y_tw,z_tw, c =load_steps, cmap = 'viridis', marker = "^", alpha = 0.5,label = f"Twin_{tw_n}")

pg_patch = mlines.Line2D([],[], marker = 'o', linestyle='None', color = "black", label = "PG") # [],[] these are just to represent empty "x" data and "y" data
tw_patch = mlines.Line2D([],[],marker = '^', linestyle='None',color = "black",label = f"Twin-{tw_n}")
ax.legend(handles = [tw_patch, pg_patch],frameon = False,loc='upper right',fontsize=15) # for: pg-and twin both
ax.set_xlabel(r'$x$ ($\mu m$)',labelpad=20)
ax.set_ylabel(r'$y$ ($\mu m$)',labelpad=20)
ax.set_zlabel(r'$z$ ($\mu m$)',labelpad=20, rotation=90)
ax.set_xlim([min(x_tw)-100, max(x_tw)+100])
ax.set_ylim([min(y_tw)-100, max(y_tw)+100])
ax.set_zlim([min(z_tw)-100, max(z_tw)+100])
# ax.set_xlim([-350, 0])
# ax.set_ylim([400, 700])
# ax.set_zlim([-1100, -600])

# Create a ScalarMappable for the colorbar
norm = Normalize(vmin=min(load_steps), vmax=max(load_steps))
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.125, shrink=0.5)
cbar.set_label('Load Range', rotation=90, labelpad=15)
# Set colorbar ticks
cbar.set_ticks(load_steps)
cbar.set_ticklabels(load_steps)

ax.grid(True)
plt.savefig(f'{pth}/sample_{sam_n}/images/twins_track/trace_pg_twin{tw_n}_pair_load{load_s}_{load_f}_misoritol{twin_tol}_radpgtol{pg_rad_tol}_avgradf{avg_rad_factor}_{approach}_3D.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
# plt.show()
plt.close()

# '''





