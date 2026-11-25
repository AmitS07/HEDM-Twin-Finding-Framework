'''
Date: 11th Nov. 2024
Author: Amit, ACDM lab
Descripson: This script does the Phase, microstructure analysis from EBSD data and XRD data

'''
#################################################################################################
#+++++++++++++++ Part-0: Importing required libaries and functions ++++++++++++++++++#
#################################################################################################
#%% 0.1 Python libraries
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
#%% 0.2 Plotting requirement 
fontsize = 13
plt.rcParams.update({"font.size":fontsize, "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,})
# plt.rcParams.update({"font.size":25})
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = [4,4]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

###################################################################################################################
#++++++++++++++++++ Part-1(XRD-analysis): Estimating 2theta diffraction angle for Co taget from the Cu-source ++++++++++++++++++#
####################################################################################################################
''' 
cu_angles = [35.2,38.3,40.33,53.09,63.35,70.7,74.65,76.55,77.5] #%% Diffrcation 2theta angle for Cu_source
lamd1 = 1.54 # vavelength of Cu-K_alpha radiation
lamd2 = 1.79 # Wavelength of Co-K_alpha radiation
co_angles = [] #%% Diffrcation 2theta angle for Co_source
for a1 in cu_angles:
    theta1 = a1*(2*pi/360) # Radian
    theta1 = np.sin(theta1/2) #numpy takes angles only in radians, It for Cu
    theta2 = np.arcsin((lamd2*theta1)/lamd1)
    theta2 = theta2*(360/(2*pi))
#     print(a2*2)
    co_angles.append(theta2*2)
print(f"Ti XRD peak postion for Co target:", np.round(co_angles,3))
'''
#################################################################################################
#++++++++++++++++++++++++++++++++++++++ Part-2(XRD-analysis-continue): Phase analysis from XRD data ++++++++++++++++++#
#################################################################################################
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
sam_n = eval(input("Enter the sample number:"))
print(sam_n)
''' 
#%% 2.1: Reading directory and loading path
# xrd_data = pd.read_fwf(f"{pth}/sample_{sam_n}/XRD_data/Ti_sample2.txt", delimiter = "\t ", header = None) # fwf =fixed-width format
xrd_data = pd.read_csv(f"{pth}/sample_{sam_n}/XRD_data/CP Ti_no puck.ASC", delimiter = " ", header = None) #
twotheta = xrd_data.iloc[:,0]
intensity_raw = xrd_data.iloc[:,1]
# intensity_raw2 = xrd_data.iloc[:,2]
intensity_smooth = savgol_filter(intensity_raw, window_length=30, polyorder=2)
#%% 2.2: Plotting data
plt.plot(twotheta, intensity_raw, color = "blue", linewidth = 1, alpha= 0.3, label = "Raw data")
plt.plot(twotheta, intensity_smooth, color = "green", linewidth = 1, label = "Smooth data")
#%%--------------------------------------------------------------------------------------------
#%% Indexing peak, This need to be added manually
plt.text(35.18, 230, r'$(10\bar{1}0)$', fontsize=11, color='k', rotation=90, ha='center', va='bottom')
plt.text(38.3, 760, r'$(0002)$', fontsize=11, color='k', rotation=90, ha='center', va='bottom')
plt.text(40.27, 1207, r'$(10\bar{1}1)$', fontsize=11, color='k', rotation=90, ha='center', va='bottom')
plt.text(53.09, 665, r'$(10\bar{1}2)$', fontsize=11, color='k', rotation=90, ha='center', va='bottom')
plt.text(63.05, 810, r'$(11\bar{2}0)$', fontsize=11, color='k', rotation=90, ha='center', va='bottom')
plt.text(70.7, 450, r'$(10\bar{1}3)$', fontsize=11, color='k', rotation=90, ha='center', va='bottom')
plt.text(76.35, 458, r'$(20\bar{2}0)$', fontsize=11, color='k', rotation=90, ha='center', va='bottom')
#%%--------------------------------------------------------------------------------------------
x_ticks = [10, 20, 30,40,50,60,70,80,90,100]
plt.xticks(x_ticks)
# plt.xlim(25,90)
plt.ylim(0, 1400)
plt.xlabel(r"2$\theta$ ($^\circ$)")
plt.ylabel(r"Intensity $(a.u.)$")
plt.legend(frameon = False, loc = "upper right")
plt.savefig(f"{pth}/sample_{sam_n}/XRD_data/theta_intensity_plot_Cu_target2.png",dpi=300,bbox_inches='tight',pad_inches=0.55)
'''
####################################################################################################################
#++++++++++++++++++++++++++++++++++++++ Part-3(Tensile analysis): As-tested load Vs displacemnt ++++++++++++++++++#
#####################################################################################################################
## Analysis of Measured applied laod vs displacement data
applied_data = pd.read_csv(f"{pth}/sample_{sam_n}/as_tested_data/load_disp_asreceived.csv", header=0)
area_0 = 1*1 # Initail-cross-section area (mm**2)
len_0 = 8 # Initial length (mm), this is as per log-file or inital sample dimension.It also matches with length infrom Jun's paper
#%% Data as tested
disp = applied_data.loc[:]["Disp"].tolist()
load = applied_data.loc[:]["Load"].tolist()
# strain = [x/len_0 for x in disp]
# stress = [x/area_0 for x in load]
#%% Plot
fig = plt.figure(figsize=(4, 4))
ax =  fig.add_subplot()
ax.set_box_aspect(1)
ax.plot(disp, load, linestyle = "-", linewidth = 1, color = "grey") #marker = "D", label = "As-tested data"
#***************************************************************************************************************************
#%% Data analysed in Twinning
analyzed_data = pd.read_csv(f"{pth}/sample_{sam_n}/as_tested_data/load_disp_as_scanned.txt", sep=" ", header = 0)
disp_anl = analyzed_data.iloc[:,0].tolist()
load_anl = analyzed_data.iloc[:,1].tolist()
# stress_anl = [x/area_0 for x in load_anl]
# strain_anl = [x/len_0 for x in disp_anl]
ax.scatter(disp_anl[:-2], load_anl[:-2], marker = "x", s= 30, color = "k", zorder = 2) #marker = "o", , label = "Analyzed data"
#***************************************************************************************************************************
#%% For Load vs Disp
# ax.legend(frameon = False, loc = "upper right")
x_ticks = np.arange(0,0.6,0.1)
ax.set_xticks(x_ticks)
ax.set_xlim(0, 0.5)
ax.set_xlabel(f"Displacement $(mm)$")
y_ticks = np.arange(0,500,100)
ax.set_yticks(y_ticks)
ax.set_ylim(0,400)
ax.set_ylabel(f"Applied load $(N)$")
plt.savefig(f"{pth}/sample_{sam_n}/as_tested_data/load_vs_disp",dpi=300,bbox_inches='tight',pad_inches=0.55)
plt.close()
#***************************************************************************************************************************
##%% for Stress vs Strain
# plt.plot(strain, stress, linestyle = "-", color = "grey", marker = "D", label = "as-tested at APS")
# plt.plot(strain_anl, stress_anl, linestyle = "--", color = "k", marker = "o", label = "data analyzed for twinning")
# plt.legend(frameon = False, loc = "upper right")
# plt.xlim(0, 0.06)
# plt.ylim(0,450)
# plt.xlabel(r"$\varepsilon_{app.}$")
# plt.ylabel(r"$\sigma_{app.}$ $(MPa)$")
# plt.xlabel(r"Eng. Strain")
# plt.ylabel(r"Eng. Stress $(MPa)$")
# plt.savefig(f"{pth}/sample_{sam_n}/as_tested_data/Eng_stress_strain_as_tested1",dpi=300,bbox_inches='tight',pad_inches=0.55)
# plt.close()