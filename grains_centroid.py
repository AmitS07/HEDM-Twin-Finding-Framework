"""
Created on Thr Aug 31 2023

@author: asingh38
     2. Caution: while selecting "Z-offset" for each type of sample, For sample#1, Z-Offset has to added as-
                Sample#1:: Layer-1# Z+0, Layer-2: Z+(Z_offset}, Layer-3: Z+(2*Z_offset)
                Sample#2:: Layer-1# Z+0, Layer-2: Z+(Z_offset}, Layer-3: Z+(2*Z_offset)
                Sample#3:: Layer-1: Z+Z_offset, Layer-2: Z+0, Layer-3: Z+(2*Z_offset)
"""
#%% Importing libraries
import numpy as np
import pandas as pd
import math
import os
import glob
import statistics
import matplotlib.pyplot as plt
#####################################################################
#%% Ploting and text format
fontsize = 13
plt.rcParams.update({"font.size":fontsize, "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,})
# plt.rcParams.update({"font.size":25})
plt.rcParams["text.usetex"] = True
# plt.rcParams["figure.figsize"] = [7,7]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#####################################################################
#%% Defining path and directories to the data-set
pth = '/home/asingh38/projects/nsf_twinning_2022'
sam_n = eval(input("Enter sample number:")) #3 
z_offset = eval(input("Enter Z-offset:")) #200
load_c = eval(input("Enter load at investigation:")) #200

#%% Markers, legend and colors
mark = ['H','s','d']
legend = ['Layer-1', 'Layer-2', 'Layer-3']
colors = ['#481567FF', '#20A387FF', '#DCE319FF']
####################################################################################
#%% Part-1:: Grains as-received in three-layer data set, individual layer-wise data
####################################################################################
''' 
# #%% Plot
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection = '3d')
ax.set_box_aspect((1, 1, 0.6)) 
#%% Path and Directories
fls_p  = glob.glob(f'{pth}/raw_data/sample_{sam_n}/parquet_files/sam{sam_n}_{load_c}N_L*.parquet')
fls_p.sort()
#%% Importing data
n_file = fls_p[0:3] # These numbers are the file index number in above "fls_p"
print(n_file)
for i,fle in enumerate(n_file):
    # fn = os.path.basename(fle) # To drop the path/directory name, only file name will give
    # splt_fls = fn.split('_')
    # sample = splt_fls[0][-1]
    # load = splt_fls[1][:-1]
    # layer = splt_fls[2][-1]
    df = pd.read_parquet(fle)
#    dat_fn = os.path.splitext(fls_p_name)[0]
    if i == 0:
        coo = df[['X','Y','Z']].values
        x = coo[:,0]
        y = coo[:,1]
        z = coo[:,2]
        z = z+z_offset
        pass
    elif i == 1:
        coo = df[['X','Y','Z']].values
        x = coo[:,0]
        y = coo[:,1]
        z = coo[:,2]
        z = z-0
    elif i == 2:
        coo = df[['X','Y','Z']].values
        x = coo[:,0]
        y = coo[:,1]
        z = coo[:,2]
        z = z-z_offset
    else:
       pass
    #### For 2D plot
    # plt.scatter(y,z, marker=mark[i], color = colors[i], edgecolor='none', s=50, alpha=0.5, label = legend[i]) # for 2D plot
    #### For 3D plot
    ax.scatter(x,y,z, marker=mark[i], color = colors[i],edgecolor='k', linewidth = 0.1, s=20, alpha=0.7) 

#for 2D plot
# plt.text(x=550,y=140,s="Layer-1", rotation=90)
# plt.text(x=550,y=-70,s="Layer-2", rotation=90)
# plt.text(x=550,y=-270,s="Layer-3", rotation=90)
# # plt.legend(frameon=False, ncols = 3, loc='upper center')
# plt.axhline(y=-100, color='k', linestyle='--',linewidth = 1) # for z-offset 400 um: y = -200
# plt.axhline(y=100, color='k', linestyle='--',linewidth = 1) # for z-offset 400 um: y = -600
# plt.xlabel(r'$y$ ($\mu m$)')
# plt.ylabel(r'$z$ ($\mu m$)')
# plt.xlim(-600,620)
# plt.ylim(-400,400)
# plt.savefig(f'{pth}/raw_data/sample_{sam_n}/images/sam{sam_n}_{int(load_c):03d}N_grains_coo_layer123_raw_yz.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
# plt.close()
#%%---------------------------------------------------------------------------------------------------
#for 3D plot
# fig.subplots_adjust(left=0.23,right=0.88,bottom=0.20,top=0.95)
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
# plt.legend(frameon=False,ncols = 3,loc='upper center') #ncols = 3,
# ax.view_init(elev=15, azim=315)
# plt.axis("equal")
# plt.show()
plt.savefig(f'{pth}/raw_data/sample_{sam_n}/images/sam{sam_n}_{int(load_c):03d}N_grains_coo_layer123_raw_3d.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()
# exit(0)
'''
########################################################################
#%% Part-2:: Grains after merging three layers data to one single layer 
########################################################################
'''
# #%% Plot tools
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection = '3d')
ax.set_box_aspect((1, 1, 0.6)) 
#%% Path and Directories
df_set = pd.read_parquet(f"{pth}/raw_data/sample_{sam_n}/parquet_files/sam{sam_n}_{load_c}N_layered_grains.parquet")
df2_t = df_set[df_set["Z"]>(z_offset/2)] # Top layer data
df2_b = df_set[df_set["Z"]<-(z_offset/2)] # Middle layer data
df2_m = df_set[(df_set["Z"]>=(-z_offset/2)) & (df_set["Z"]<=(z_offset/2))] # Bottom layer data

X1 = df2_t.loc[:, "X"].tolist()
Y1 = df2_t.loc[:, "Y"].tolist()
Z1 = df2_t.loc[:, "Z"].tolist()
X2 = df2_m.loc[:, "X"].tolist()
Y2 = df2_m.loc[:, "Y"].tolist()
Z2 = df2_m.loc[:, "Z"].tolist()
X3 = df2_b.loc[:, "X"].tolist()
Y3 = df2_b.loc[:, "Y"].tolist()
Z3 = df2_b.loc[:, "Z"].tolist()

##************************2D plot ***********************
# plt.scatter(X1, Z1, marker = mark[0], color = 'gray', s = 50, edgecolor='none', alpha = 0.5, label = 'Layer:1') #mark[0]
# plt.scatter(X2, Z2, marker = mark[1], color = 'gray', s = 50, edgecolor='none', alpha = 0.5, label = 'Layer:2') #colors[1]
# plt.scatter(X3, Z3, marker = mark[2], color = 'gray', s = 50, edgecolor='none', alpha = 0.5, label = 'Layer:3')
# # plt.text(x=530,y=140,s="Layer-1", rotation=90)
# # plt.text(x=530,y=-70,s="Layer-2", rotation=90)
# # plt.text(x=530,y=-270,s="Layer-3", rotation=90)
# # plt.legend(frameon=False, ncols = 3, loc='upper center')
# plt.axhline(y=-100, color='k', linestyle='--',linewidth = 1) # for z-offset 400 um: y = -200
# plt.axhline(y=100, color='k', linestyle='--',linewidth = 1) # for z-offset 400 um: y = -600
# plt.xlabel(r'$x$ ($\mu m$)')
# plt.ylabel(r'$z$ ($\mu m$)')
# plt.xlim(-600,600)
# plt.ylim(-400,400)
# plt.savefig(f'{pth}/raw_data/sample_{sam_n}/images/sam{sam_n}_{int(load_c):03d}N_grains_coo_merged_layer_xz.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
# plt.close()
##************************3D plot ***********************
ax.scatter(X1, Y1, Z1, marker='o', s=20, color = 'gray',edgecolor='k', linewidth = 0.1, alpha = 0.7) #colors[0],label = 'Layer:1'
ax.scatter(X2, Y2, Z2, marker='o', s=20, color = 'gray',edgecolor='k', linewidth = 0.1, alpha= 0.7)
ax.scatter(X3, Y3, Z3, marker='o', s=20, color = 'gray',edgecolor='k', linewidth = 0.1, alpha= 0.7)
# fig.subplots_adjust(left=0.23,right=0.88,bottom=0.20,top=0.95)
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
plt.savefig(f'{pth}/raw_data/sample_{sam_n}/images/sam{sam_n}_{int(load_c):03d}N_grains_coo_merged_layer_3d.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()
exit(0)
'''
################################################################################################
#%% Part-3:: Grains after matching across all load steps as final matched grains across one layer
#################################################################################################
# ''' 
#%% Loading load-wise final matched grains-Ids of the applied load
load_s, load_f = 0, 320
final_all_matchgrainids = pd.read_csv(f'{pth}/raw_data/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv', sep ="\t").dropna()
final_grainids_load = final_all_matchgrainids[f"Load_{load_c}"]
#%% Reading grains data at applied load
df_final_matchgrains = pd.read_csv(f"{pth}/raw_data/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
coo_final_grains = df_final_matchgrains.loc[final_grainids_load,"X":"Z"]
x_coo_final = coo_final_grains["X"]
y_coo_final = coo_final_grains["Y"]
z_coo_final = coo_final_grains["Z"]
#************************************ for 3D plot ************************************
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection = '3d')
ax.set_box_aspect((1, 1, 0.6)) 
ax.scatter(x_coo_final,y_coo_final,z_coo_final, marker='o', s=20, color = 'gray',edgecolor='k', linewidth = 0.1, alpha = 0.7) 
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
plt.savefig(f'{pth}/raw_data/sample_{sam_n}/images/sam{sam_n}_{int(load_c):03d}N_finalgrains_coo_merged_layer_3d.png', dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()
exit(0)
#'''
################################################################################################
#%% Part-4:: Grains of  just one single layer 
#################################################################################################
'''
#%% Path and Directories
df = pd.read_parquet(f"{pth}/raw_data/sample_{sam_n}/parquet_files/sam{sam_n}_{load_c}N_layered_grains.parquet")
Z = df.loc[:, "Z"].tolist()
X = df.loc[:, "X"].tolist()
##************************2D plot ***********************
plt.scatter(X, Z,marker = "D",color = "tab:grey", alpha= 0.5)
'''
