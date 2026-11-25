''' 
Date: 7th April 2025
Author: Amit, ACDM Lab, Email: amitdravid@outlook.com, asingh38@crimson.ua.edu
Description: This python code does calculates the following from the final-considred(matched) grains HEDM data -
            (1): Dimensions of the measured volume at particluar load
            (2): Radius scaling factor (assumed spherical), which is estimated from the volumentric ration of the measured domain volume over volume of all grains in domain
            (3): Boolean index of grains
Note: 1. Care must be taken on setting the plot's x and y limits

'''
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Part-0: Importing Libraries from python and defing the path to directories
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
# from mpl_toolkits.axes_grid1 import make_axes_locatable
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

#%%Plotting requirments
#%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%
#%Ploting requirment
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
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Importing variables and defining path and directories
sam_n = eval(input("Enter sample number:"))
load_range = list(map(int, input(f"Enter all loads (space seprated) being analyzed: ").split()))
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-1: Extracting and saving grain's details e.g. coordinates, radius/diameters, orientations
#%% Note: This shouldbe run once, just to save the data
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% 1.1: Path and directories
pth = "/home/asingh38/projects/nsf_twinning_2022/raw_data"
#1.2: Based on twin-IDs of "pg_twins-dict", findout corresponding "extra-matched grains" of unmatched grains for the load-range of k(th)-n(th)
#1.2.1: Reading the list of matched grains from list of "UNMATHCED" grains i.e. extra-grains
matchgrainIDs_um = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_fm_unmatched_loadwise/final_common_grains_loadwise_{load_s}_{load_f}.csv",sep ="\t").dropna()
#1.2.2 Reading list of final-matched grains i.e. untimately matched grains across all load steps from kth-nth load step
final_matchgrainIDs = pd.read_csv(f"{pth}/sample_{sam_n}/matched_grains_loadwise/final_common_grains_loadwise_0_{load_f}.csv",sep ="\t").dropna()
load_twin = int(input("Enter load at which neighbors-grains being analyze:"))
'''
load_select = [load_nghbr]
for load_c in load_select:
    cord_allpg, rad_allpg, dia_allpg, oris_allpg = [], [], [], []
    for ii in final_matchgrainIDs[f"Load_{load_c}"]:
        ##% Reading entire data_frames for all the grains
        df_pgs_all = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv",skiprows=1, sep = "\t").set_index(["%new_GrainID"])
        ##% --------------------------------------------------------------------------------------------------------------------------------------------
        ##% Note: The divison of 1000 is done because the domian of RVE is scaled wrt to 1 unit, i.e 1000 = 1 unit in Neper 
        cord_allpg.append(np.array(df_pgs_all.loc[ii]["X":"Z"])/1000) #Grain's Position
        rad_allpg.append(df_pgs_all.loc[ii]["GrainRadius"]*(1/1000)) #Grain's Radius
        dia_allpg.append(df_pgs_all.loc[ii]["GrainRadius"]*(2/1000)) #Grain's Diameter 
        oris_allpg.append(np.array(df_pgs_all.loc[ii]["O11":"O33"]))#Grain's Orientation
    ##%Saving the "cord" and "diameter" data
    np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/centroidals_load{load_c}.txt", cord_allpg, delimiter = "\t",fmt ="%.10f")
    np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/radius_load{load_c}.txt", rad_allpg, delimiter = "\t", fmt = "%.10f")
    np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/diameters_load{load_c}.txt", dia_allpg, delimiter = "\t", fmt = "%.10f")
    np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/orientations_load{load_c}.txt", oris_allpg, delimiter = "\t", fmt = "%.10f")
##% Saving centroidal and diameter files combined together
cord_dia_allpg = np.hstack((np.array(cord_allpg), np.array(dia_allpg).reshape(-1,1)))
np.savetxt(f"{pth}/sample_{sam_n}/neighbour_grains/centroidals_dia_load{load_twin}.txt", cord_dia_allpg, delimiter = "\t",fmt ="%.10f")
'''
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% Part-2: HEDM measured domian dimensions calculations
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# '''
#%% 2.1: Importing saved grain's diameters
dia_hedm = np.loadtxt(f"{pth}/sample_{sam_n}/neighbour_grains/diameters_load{load_twin}.txt")
#%% 2.2: Importing saved 
cord_hedm = np.loadtxt(f"{pth}/sample_{sam_n}/neighbour_grains/centroidals_load{load_twin}.txt")
x_ld, y_ld, z_ld = np.array(cord_hedm)[:,0], np.array(cord_hedm)[:,1], np.array(cord_hedm)[:,2]
w = (max(x_ld)-min(x_ld))
t = (max(y_ld)-min(y_ld))
h = (max(z_ld)-min(z_ld))
print(f"Domain dimensions(w,t,h):{w,t,h}")
vol_domain = w*t*h
print(f"Volume of the measured domain: {vol_domain}")
vol_allpg = [(np.pi/6)*(d**3) for d in dia_hedm]
total_vol_allpg = np.sum(vol_allpg)
#%% Volume-factor calculations
v_f= vol_domain/total_vol_allpg
#%-----------------------------------------------------------------------
#%% Coorected Radius-factor, asumming sphereical
r_f = v_f**(1/3)
print(f"Radius-factor: {r_f}")
exit(0)
# '''
##%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##% Part-3: Statistical comparison of grains from FF-HEDM and Neper 
##% Note: 1. Before runnning this part of the code, one must have "stcell" data from Neper after running script "neighbor_grains_neper.sh"
##% 2. All statistical calculations/plots are done in the same scale as these are beaing measured using 3D HEDM i.e. in 1000 
##%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##% 3.1: Reading NEPER generated grain's diameters, and scaling up by 1000
dia_hedm = np.array(dia_hedm)*1000 #scaling up by 1000, since previosuly it was scaled down by 1000. 
mu  = eval(input(f"Enter the mean of grain'size sphericity distribution:")) #0.145
sigma = eval(input(f"Enter the standard-dev of grain'size sphericity distribution:")) #0.03
dia_sf_neper = eval(input(f"Enter the scale-up-factor of NEPER:")) # diameter scale-factor = 2.112480, by which NEPER scale the size of diameter of the grains to fill the volume of domian
wt_method = input(f"Enter weight-method('rad'):")
# '''
##% 3.2: Comparisons of grains diameters between the HEDM and NEPER
np_dia_file = f"diameq_nghbrs_centdia_mu{mu}_sig{sigma}_w{wt_method}_load{load_twin}"
#%% Importing NEPER generated grain's diameters
dia_tess = np.loadtxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/{np_dia_file}.stcell")
dia_neper = np.array(dia_tess)*1000/(dia_sf_neper)
#%%----------------------------------------------------------------------------------------
#%%----------------------------------------------------------------------------------------
#%% 3.3 Plotting simple scatter plot with out density
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
# ax.set_facecolor('white')
ax.set_position([0.15,0.20,0.60,0.65])  #main plot to occupy an exact coordinate box: [left, bottom, width, height], percentages of the figure (0.0 to 1.0)
ax.set_box_aspect(1)
#%% 3.3.2 To plot the diagonal 1-1 line
xymax = np.ceil(max(max(dia_hedm),max(dia_neper))/10)*10
ax.plot([0,xymax], [0,xymax], linewidth = 1, color = "black",linestyle = "--",alpha = 0.7) # To put diagonal line in graph

#%% 3.3.1 Plotting simple scatter plot with out density
''' 
# ax.scatter(dia_hedm, dia_neper, marker = "o", color = "white", edgecolor='black',alpha = 0.5)
# ax.set_xlim(-10,xymax)
# ax.set_ylim(-10,xymax)
# ax.set_xlabel(r"$d$ ($\mu m$)")
# ax.set_ylabel(r"$d^{*}$($\mu m$)")
# plt.savefig(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/dia_compare_{np_dia_file}_n.png", dpi=300,bbox_inches='tight',pad_inches=0.55)
'''
#%% 3.3.2 Probabability density plots
#%% ************************* Jerry approach ***************************
# bin_size = 5
# xbins = np.arange(0,xymax, bin_size)
# ybins = np.arange(0,xymax, bin_size)
# counts, xedges, yedges = np.histogram2d(dia_hedm, dia_neper, bins = [xbins, ybins])
# #%% Bin-center coordinates
# Xc = 0.5 * (xedges[:-1] + xedges[1:])
# Yc = 0.5 * (yedges[:-1] + yedges[1:])
# Xc, Yc = np.meshgrid(Xc, Yc)
# #%% Count field
# Z_count = counts.T.astype(float)
# #%% Smooth the underlying plotting field
# Z_smooth = gaussian_filter(Z_count,sigma=1)
# white_threshold = 1
# Z_density = np.ma.masked_where(Z_smooth < white_threshold,Z_smooth)
# #%% Plot
# colorbar_min, colorbar_max = 0, 85
# fill_levels = np.linspace(colorbar_min,colorbar_max,10)
# contour_fill = ax.contourf(Xc,Yc,Z_density,levels=fill_levels,cmap="viridis")
# #%% Color bar
# cbar_ticks = [0, 20, 40, 60, 80,100]
# cbar = fig.colorbar(contour_fill,ax=ax,fraction=0.046,pad=0.04,ticks=cbar_ticks)
# cbar.ax.set_title("Counts", pad=20)
#%% ************************* Approach-II ***************************
xy = np.vstack([dia_hedm, dia_neper])
z = gaussian_kde(xy)(xy)
z_scaled = z*100
#%% sort points by density
idx = z_scaled.argsort()
x,y, z_scaled = dia_hedm[idx], dia_neper[idx], z_scaled[idx]
sc = ax.scatter(x,y,c=z_scaled,s=5, marker = 'o', cmap = 'viridis')

cax = fig.add_axes([0.80, 0.225, 0.04, 0.601])
cbar = fig.colorbar(sc,cax=cax) #, shrink = 0.817, pad = 0.04
# Manually customize the cbar ticks and format them
ticks_layout = np.linspace(0.0, z_scaled.max(), 5) 
cbar.set_ticks(ticks_layout)
cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks_layout])
cbar.set_label('Density', labelpad = 5)

#%% ticks, labels and limits
ax.set_xlim(0,xymax)
ax.set_ylim(0,xymax)
ticks = np.arange(0, 110, 20)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r"$d$ ($\mu m$)")
ax.set_ylabel(r"$d^{*}$($\mu m$)")
plt.savefig(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/dia_compare_{np_dia_file}_dp.png", dpi=300,pad_inches=0.55) #bbox_inches='tight'
plt.close()
# exit(0)
# ''' 
##%-----------------------------------------------------------------------------------------------------------------------------------------
##% 3.4: Comparison of grain's centroidal between the HEDM and NEPER
np_coo_file =  f"coo_nghbrs_centdia_mu{mu}_sig{sigma}_w{wt_method}_load{load_twin}"
cord_neper = np.loadtxt(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/{np_coo_file}.stcell")
## Euclidean dist
dist = []
for i,j in zip(cord_hedm,cord_neper):
    dist.append(euclidean_dist(i,j))
euclid_dist = np.array(dist)

## To normalize the eucliden distance, the HEDM measured grains dia meter was used-
rad_hedm = np.loadtxt(f"{pth}/sample_{sam_n}/neighbour_grains/radius_load{load_twin}.txt")
norm_dist = [x/y for x,y in zip(euclid_dist,rad_hedm)]
## Mean and standard-distribution
# mu_euclid = np.mean(euclid_dist)
# sigma_euclid =np.std(euclid_dist)
# print(f"Euclid:: Mean: {mu_euclid:0.3f}, Std:{sigma_euclid:0.3f}")
mu_eucld_norm = np.mean(norm_dist)
sigma_eucld_norm =np.std(norm_dist)
print(f"Norm-euclid:: Mean: {mu_eucld_norm:0.3f}, Std:{sigma_eucld_norm:0.3f}")

##% 3.5 Plotting
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
ax.set_position([0.30,0.20,0.60,0.65]) 
ax.set_box_aspect(1)
#%%************************************************************************************ 
#%% To fit PDFs with Gaussina kde
#% with out normalization
# x_data = np.linspace(min(euclid_dist), max(euclid_dist), 500)
# kde_fit = stats.gaussian_kde(euclid_dist)
#%% With normalized data
# x_data = np.linspace(min(norm_dist), max(norm_dist), 500)
# kde_fit = stats.gaussian_kde(norm_dist)
# ax.plot(x_data, kde_fit(x_data), linestyle='-', color = "k") #label="Kernel-Density"
#%%************************************************************************************ 
#%% Histogram plot
# ax.hist(euclid_dist, bins = 30,density  = True, linewidth = 1.5,color = "white", edgecolor='black', alpha = 0.7)#histtype = "step"
ax.hist(norm_dist, bins = 20,density  = True, linewidth = 0.5, color = "gray", edgecolor='black', alpha = 0.7)#histtype = "step"
#%%************************************************************************************   
# plt.legend(frameon = False, loc = "upper right")
ax.set_xticks([0,1,2,3,4,5])
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
# ax.set_aspect(1.0/ax.get_data_ratio()) # To make square plot
ax.set_xlim(0,5) 
ax.set_ylim(0,1.2)
ax.set_xlabel(r'$\hat{\delta}$') #($\frac{euclid_{i}}{dia_{i}}$)
ax.set_ylabel(r'Frequency') #r'Probability density'
# plt.show()
plt.savefig(f"{pth}/sample_{sam_n}/neighbour_grains/tess/stcell/norm_euclid_{np_coo_file}_bar.png",  dpi=300,pad_inches=0.55) #,bbox_inches='tight'
plt.close()

