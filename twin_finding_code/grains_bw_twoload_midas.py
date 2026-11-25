'''
Created: 29th Feb 2024
Author: Amit Singh, ACDM (Advanced Computational Deformation Mechanics) Lab 
Descripson: This code will do the grains-matching between load steps, after running from MIDAS
            (i.e. after having misorientation and eucliden-dist data), and then
            merged the common matched grains between the two based on Ids
Modified date: 21st August 2025, for data set which was not taken at constant load increment.
'''
# importing libaries
# from in-build python
import os
import glob
import sys
import pandas as pd
import numpy as np
import time 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %%
## This is added to call the functions libraries
lib_path = os.path.expanduser('~/codes/nsf_twinning_2022')
# Append it to sys.path if not already there
if lib_path not in sys.path:
    sys.path.append(lib_path)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing library: maunalu defined
from lib_grains_matching_load_midas import *

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++ Part:0 Defining path to folders and entering input variables  +++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

pth = '/home/asingh38/projects/nsf_twinning_2022/raw_data'
sam_n = eval(input("Enter the sample number:")) #%% Sample number
load_range = list(map(int, input(f"Enter all loads (space seprated): ").split()))
print(f"Load-Range:", load_range)
#----------------- Part:1.1 Parameters used in running MIDAS for load across all load steps --------------#
z_offset = eval(input("Enter the Z_offset:")) # =0, enter the Z-offset iff you have not corrected it.
beamThickness = eval(input("Enter the Beamthickness:")) # =1200 ## It is beamheight, i.e z-offset bw each load steps
twinAngle = eval(input("Twin-angle(0{not considering twin} else 85)(in deg):")) # =0 
tol_misori = eval(input("Misori-tolerance(in deg):")) #0.5 in degree, for sample-3 its 1
tol_euclid = eval(input("Enter Euclidian-dist tolerance(um):")) #50 in um
load_s = load_range[0]
load_f = load_range[-1]
print(f"--"*20)
print(f"Starting applied-load: {load_s}")
print(f"Final applied-load: {load_f}")
print(25*"+++")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++ Part-1: Begin MIDAS, to save the 'misor-euclid-grainsIDS between two consecutive load-steps  +++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# '''
#This section of the code is save the file which is MIDAS readable
for i in range(0, len(load_range),1):
    tic1 = time.time()
    load_i = load_range[i]
    load_ii = load_range[i+1]
    #%%%%%%%%%%%%%%%%%% Mathcing Grains loadwise: Running MIDAS %%%%%%%%%%%%%%%%%%%%%%#
    # Required file name
    OutFile = os.path.join(f'{pth}/sample_{sam_n}/matched_grains_loadwise/midas_output/sam{sam_n}_load{load_i}_{load_ii}_twin{twinAngle}_beamheight{beamThickness}.csv')
    InputFile1 = glob.glob(f'{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_i}N_grains.csv')
    InputFile2 = glob.glob(f'{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_ii}N_grains.csv')  
    # For printing number rof grains
    df1 = pd.read_csv(InputFile1[0])
    df2 = pd.read_csv(InputFile2[0])      
    print(f"# Grains in input data {load_i}:", len(df1)-1)
    print(f"# Grains in input data {load_ii}:" ,len(df2)-1)

    #%% MIDAS Run 
    MIDAS_grains_match(z_offset, beamThickness, twinAngle, OutFile, InputFile1, InputFile2)
    tic2 = time.time()
    print(f"Total Time elapsed: {(tic2-tic1):.3f} s")
    print(25*"+++") 
    ## To Break the load, when two consecutive load-steps used
    if load_ii == load_f:
        break

# ''' 