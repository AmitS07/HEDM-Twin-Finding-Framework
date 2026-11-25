'''
Date: 2nd August 2024
Author: Amit, ACDM Lab
Description: This script calculate the volume fraction differences between the measured/scanned( with Sychrotron)
            layer/s of given sample to the volume occupied by all the grains in that particular layer/s

Note: This script work if the HEDM scan were done with constant domain height or varied height measuremnet. But, if the 
        it worked if the measured height were chnaged only once at any load-step. However the if loop could be further extended 
        for if measurement were done with multiple different heights
'''
# Loading libraries
import pandas as pd
import numpy as np

#%% Deining Path and load-ranges
sam_n = eval(input(" Enter sample number:"))
pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
load_range = list(map(int, input(f"Enter all loads (space seprated) being analyzed: ").split()))
load_s = load_range[0]
load_f = load_range[-1]
print(f"Starting load for analysis: {load_s}")
print(f"Final load for analysis: {load_f}")
#%% Check if the measurement was done for all load steps with same Domain
domain_height = input("Is domian-height same across all load (type 'yes' or 'no'?):")
load_d = input("If 'no', enter the load from which scanned height changes:")
l = eval(input(f"Enter the length (um) of scanned-layer:")) #1000 um = 1mm
w = eval(input(f"Enter the width (um) of scanned-layer:")) #1000 um = 1mm
#%% Estimating height
if domain_height=='no':
    h = eval(input(f"Enter the height(um) of Scanned-layer:")) # Beased on the beam height
    h_d = eval(input(f"Enter the height(um) of Scanned-layer after load {load_d}:")) # Beased on the beam height
else:
    #%% Volume scanned with same domain height
    h = eval(input(f"Enter the height(um) of Scanned-layer:")) # Beased on the beam height
#%%******************* Calculating Volume Occupied ********************
V_f_all = []
rad_f_all = []
load_c = load_s
for load_c in load_range:
    #Loading data
    df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv",skiprows=1, sep = "\t")
    rad_grain = df_modif.loc[:]["GrainRadius"] # All the grains[:]
    V_grains = (4/3)*np.pi*(rad_grain**3)
    V_ocp = np.sum(V_grains) #V_occupied
    #%% ********************Calculateing Volume Scanned*******************
    if domain_height=='no':
        if load_c < float(load_d):
            V_measure = l*w*h
        else:
            V_measure = l*w*h_d
    else:
        V_measure = l*w*h
    #%% *************** Volume fraction factor calculation *****************
    #Volume_fraction factor
    V_f = V_measure/V_ocp
    V_f_all.append(V_f)
    #Radius factor by assuming measured-volume and all grains-volume is equivalent to sphere
    rad_f = V_f**(1/3)
    rad_f_all.append(rad_f)
    print(f"At load-{load_c}, the Volume-factor:{V_f:0.4f} || Radius-factor:{rad_f:0.4f}")
##################################################
V_f_global = np.mean(V_f_all)
rad_f_global = np.mean(rad_f_all)
print(f"The global(mean for all load-steps):: Volume factor:{V_f_global:0.4f}|| Radius factor:{rad_f_global:0.4f}")