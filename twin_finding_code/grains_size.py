#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:16:54 2022

@author: asingh38
"""
#%% Importing libraries
import numpy as np
import pandas as pd
import math
import os
import glob
import statistics
import matplotlib.pyplot as plt

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

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Part-1: Ploting histogram of grain size of the layer-wise matched
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%% Analysis of the grain morphology i.e diameter and  sphereicity
'''
#pth = os.getcwd()
pth = '/home/asingh38/projects/nsf_twinning_2022'
sam_n = '3'
load = '155'
fls_p  = glob.glob(f'{pth}/raw_data/sample_{sam_n}/parquet_files/sam{sam_n}_{load}*.parquet')
fls_p.sort()

fls_p_name = fls_p[-1]
print(fls_p_name)
data = pd.read_parquet(fls_p_name)
print(f"Number of grains: {len(data)}")
#data = data.set_index('new_GrainID')
#2.3 reading files for "GrainRadius"
#radius = data.loc[data.index[:], ['GrainRadius']]
radius = data['GrainRadius']
diameter = 2*radius
plt.hist(diameter,edgecolor='black',color = 'white',bins=np.arange(diameter.min(),diameter.max())) #density =True
#plt. axvline(diameter.mean(), color = 'black', linestyle= 'dashed', linewidth=1)
mean = np.mean(diameter)
print(mean)
#plt.text(50,200,'Mean = ', mean)
# plt.text(85, 910, f'Mean = {mean:.2f} ($\mu m$)') 
# plt.text(85,850,f'Deformed ({load} ($N$))')
plt.xlim([0, 150])
plt.ylim([0,500])
plt.xlabel(r'Grain Diameter ($\mu m$)')
plt.ylabel(r'Number of Grains')
#plt.show()
plt.savefig(f'{pth}/raw_data/sample_{sam_n}/images/Sample_{sam_n}_diameter_deformed_{load}NLoad.png') 
''' 

'''
### for plotting grains sphericity
#diameter = diameter.values.flatten()
#nrmdia= diameter/(statistics.mean(diameter)) # normalizing the diameter
sprcity = 1-grain.iloc[:, 1] # this is 1-sphericity
plt.hist(sprcity,edgecolor='black',color = 'white',bins=np.arange(min(sprcity),max(sprcity)+0.0125,0.0125)) #density =True
plt.xlim([0.0, 0.35])
plt.ylim([0,200])
plt.xlabel(r'1-s')
plt.ylabel(r'Number of Grains')
#plt.subplots_adjust(left=0.110, right=0.97, top=0.975, bottom=0.125) #wspace = 0.200, hspace = 0.200 # adjust the whitw space
plt.savefig(f'{path}/caxis_1000rand_Sphericity.png') 
'''
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Part-2: Plooting histogram of grain size of the finally tracked grains across all load steps
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pth = '/home/asingh38/projects/nsf_twinning_2022'
sam_n = eval(input(f"Enter Sample number: "))
load_c = eval(input(f"Enter load(N): "))
data_file = pd.read_csv(f"{pth}/raw_data/sample_{sam_n}/csv_files/loadwise_matched_grains/sam{sam_n}_{load_c}N_loadwise.csv", sep = "\t").set_index(["%new_GrainID"])
gr_dia = data_file["GrainRadius"]*2
print(f"Number of grains at load-{load_c}: {len(gr_dia)}")
#%% Plot
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
ax.set_position([0.30,0.20,0.60,0.65]) # It is move the white background of the figure
ax.set_box_aspect(1)
ax.hist(gr_dia, bins = 20,density  = False, linewidth = 0.5, color = "gray", edgecolor='black', alpha = 0.7)#histtype = "step"
ax.set_xticks([20,40,60,80,100])
ax.set_yticks([0,50,100,150,200,250])
ax.set_xlabel(r'$d$ $(\mu m)$') #($\frac{euclid_{i}}{dia_{i}}$)
# ax.set_ylabel(r'Frequency') #r'Probability density'
ax.set_ylabel(r'Number of grains') #r'Probability density'
# plt.show()
plt.savefig(f"{pth}/raw_data/sample_{sam_n}/images/sam{sam_n}_load{load_c}_trackedgrains_dia.png", pad_inches=0.55)


