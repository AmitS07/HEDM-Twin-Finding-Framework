'''
Created: 12th September 2025
Author: Amit Singh, ACDM (Advanced Computational Deformation Mechanics) Lab 
Descripson: This script will convert the parquet files to csv files, which were generated, 
            After layer-wise grains matching using code "grains_bw_layers.py"
'''
# importing libaries
# from in-build python
import os
import sys
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
load_s = load_range[0]
load_f = load_range[-1]
print(f"--"*20)
print(f"Starting applied-load: {load_s}")
print(f"Final applied-load: {load_f}")
print(25*"+++")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++ Part-1 Reading file and converting to .csv  +++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

##%%========================1.1: Data format/file change from Parquet to csv===============================
# ''' 
for load_c in load_range:
    #% Changing these files as per the raw data file (i.e .csv)for MIDAS run, i.e. adding headers
    parquet_to_csv_midas(pth, sam_n, load_c)
print(f"Great! Data-files are modified and can be read by MIDAS")
print(22*"+++")