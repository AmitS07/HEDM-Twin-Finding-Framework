''' 
Date: 22nd December 2025
Author: Amit, ACDM Lab, Email: amitdravid@outlook.com, asingh38@crimson.ua.edu
Description: This python code does calculate the boolean for parent and neighbor grains
Note: This script asks two types of load inputs: 'load_tess' and 'load_twin' at which tesselation is being generated, and twin form respectively. If the tessellations 
      are generated for each load in which twin might have formed then that case load_tess and load_twin could be same. 
      However, it is very obvious too that the tesselation (micorstrcure) should not change much especially with in elatsic-limit of load. Therefore, one can assume the same, 
      tessellation across all load steps. As a result one could use the same tesselaltion generated for one load. In the present study I am using tessellation generated at load 155N.
'''
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Part-0: Importing Libraries from python
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
# from scipy import stats
# import matplotlib.lines as mlines
# from matplotlib.lines import Line2D
# from matplotlib.colors import Normalize
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This is added to call the functions libraries/modules
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022') # Path to the function libraries
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from lib_twin_parent_matching import *
from lib_functions_analysis import *

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%Ploting requirment
# plt.rcParams.update({'font.size': 18})
fontsize = 13
plt.rcParams.update({"font.size":fontsize, "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,})
plt.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = [7, 7] #default: [6.4, 4.8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
# plt.rcParams['text.latex.preamble'] = [r'\boldmath']
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-1: Importing variables and defining path and directories
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% 1.1: Variables
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
consider = input(f"Enter type of analysis('all' or 'unique'):") # enter 'unique', if you want to consider no repated matched grains
compare = input("Enter PG-TW compare method(type: 'step_0' for 0th, else 'current'):") # generally 'step_o'
#%%***************************************************************************************************
#%% 1.2: Path and directories
pth = "/home/asingh38/projects/nsf_twinning_2022/raw_data"
load_twin = int(input("Enter the load for twin analyze:"))
##% 1.2.1 Reading list of final-matched grains i.e. ultimately matched grains across all load steps from kth-nth load step
final_matchgrainIDs = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_0_{load_f}.csv",sep ="\t").dropna()
final_grains_ids = final_matchgrainIDs[f"Load_{load_twin}"].tolist() # Final grains for a particular applied load

##% 1.2.2 Reading the parent-twin ids for s specific applied load
pth_tw_pg_dict = f"{pth}/sample_{sam_n}/found_twins_loadwise/from_unmatched_grains_loadstep/pg_twin_IDs_load{load_s}_{load_f}_twangle{twin_angle}_twtol{twin_tol}_radpg{pg_rad_tol}_{approach}_radf{avg_rad_factor}_{consider}.csv"
tw_pg_dict = read_csv_dictionary(pth_tw_pg_dict)
## Estimating the parent-twin pair ids for load-step at which twin formed
IDs_tw, IDs_pg = get_dictionary_values(tw_pg_dict, load_twin)
##% 1.2.3 Finding index of the parent-IDs
parent_ids = IDs_pg.tolist() # all parent IDs at that load-step

#%%1.2.4: To selected one parent-grain at once
pg_tw_pair_n= int(input("Enter parent-twin pair#: ")) #Parent-twin_pair_number, begin with 1
pg_id_pair_n = [parent_ids[pg_tw_pair_n-1]]# 1 subtracted, becaus ewe assume pair should start with number-1 instead of number-0, but Pyton read with 0
index_parent_id = [final_grains_ids.index(pg_id) for pg_id in pg_id_pair_n if pg_id in final_grains_ids]

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-2: Creating booleans 
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##% 2.1: Generating 'Boolean' only for parent grains based on the parent-ID index
bool_parent = [0]*len(final_grains_ids) # Creating a list of zeros as per same length as total number of grains i.e. id_allpg
for ii, idx in enumerate(index_parent_id):
    bool_parent[idx] = 1
## Saving boolean for parent grains only 
np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/parent/pg_pair{pg_tw_pair_n}_bool_misori{twin_angle}_twload{load_twin}.txt", bool_parent, delimiter = "\t",fmt ="%d")
##% -----------------------------------------------------------------------------------------------------------------------------------------------------
##% 2.2: Generating 'Boolean' for parent-neighbor both based on the their-ID index
load_tess =  int(input("Enter the load of tessellation:"))
mu  = eval(input(f"Enter the mean of grain'size sphericity distribution:")) #0.145
sigma = eval(input(f"Enter the standard-dev of grain'size sphericity distribution:")) #0.03
wt_method = input(f"Enter weight-method('rad'):") #rad
##1st:: Loading "neighbor_grain-index ids"
## NEPER data-set
directory = f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/neighbors"
# neighr_id_file = f"npolys_nghbrs_centdia_mu{mu}_sig{sigma}_w{wt_method}_load{load_twin}.stpoly"    
neighr_id_file = f"neighbrs_centdia_sc{mu}_ms{sigma}_load{load_tess}_w{wt_method}_npolys.stpoly"   
nghrbs_allpg_file = read_numeric_txt(f"{directory}/{neighr_id_file}") 
nghrbs_array = [nghrbs_allpg_file[ii] for ii in index_parent_id]
##2nd:: Saving the boolen for Parent-grain and corresponding neighbour
bool_parent_neigh = [0]*len(final_grains_ids)
for ii, idx in enumerate(index_parent_id):
    bool_parent_neigh[idx] = 1
    for jj in nghrbs_array[ii]:
        bool_parent_neigh[jj-1] = 1 ## The "-1" use for python, since python reads from 0, and NEPER save data from 1
## Saving boolean data for parent and neighbrs both
np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/neighbors/nghbr_bool_pg_pair{pg_tw_pair_n}_misori{twin_angle}_twload{load_twin}_w{wt_method}.txt", bool_parent_neigh, delimiter = "\t",fmt ="%d")

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-3: Testing centroidal position of the parent-neighbor grains pairs based on boolean of the index of parent-neighbours
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%3.1 Extracting neighbors-ids from list of parent-neighbor boolean
pg_ids = [final_grains_ids[ii] for ii, idx in enumerate(bool_parent) if idx == 1]
pg_ng_ids = [final_grains_ids[ii] for ii, idx in enumerate(bool_parent_neigh) if idx == 1]
ng_ids = [x for x in pg_ng_ids if x not in set(pg_ids)] # [] is enclosed to make it list, in case if there is only one parent-grain found, so that I canrun in loop
print(f'Total number of neighbors: {len(ng_ids)}')
#%%3.2: Saving parent and neighbor grains-IDs
np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/parent/pg_pair{pg_tw_pair_n}_ids_misori{twin_angle}_twload{load_twin}.txt", pg_ids, delimiter = "\t",fmt ="%d")
np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/neighbors/nghbr_ids_pg_pair{pg_tw_pair_n}_misori{twin_angle}_twload{load_twin}_w{wt_method}.txt", ng_ids, delimiter = "\t",fmt ="%d")

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-4: Visualizing centroidal positions of the parent-neighbor grains pairs based on boolean of the index of parent-neighbours
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%4.1 Loading grain's details at given twin load
df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_twin}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
#************************************ for 3D plot ************************************
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection = '3d')
ax.set_box_aspect((1, 1, 0.6)) 
#%% Plotting for parent grain
coo_pg = df_final_matchgrains.loc[pg_ids,"X":"Z"]
x_coo_pg = coo_pg["X"]
y_coo_pg = coo_pg["Y"]
z_coo_pg = coo_pg["Z"]
ax.scatter(x_coo_pg,y_coo_pg,z_coo_pg, marker='o', s=20, color = 'gray',facecolor = "none", edgecolor='k', linewidth = 0.2, alpha = 0.7)
#%% Plotting for parent+neighbor grains
coo_ng = df_final_matchgrains.loc[ng_ids,"X":"Z"]
x_coo_ng = coo_ng["X"]
y_coo_ng = coo_ng["Y"]
z_coo_ng = coo_ng["Z"]
ax.scatter(x_coo_ng,y_coo_ng,z_coo_ng, marker='*', s=20, color = 'gray',facecolor = "none", edgecolor='k', linewidth = 0.2, alpha = 0.7)

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
# plt.tight_layout()
ax.margins(0) # To reduce the gap between the axis ticks and marks
ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)
# plt.show()
plt.savefig(f'{pth}/sample_{sam_n}/images/parent_twin_pair/coo_pg_pair{pg_tw_pair_n}_nghbr_misori{twin_angle}_twload{int(load_twin):03d}.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()
