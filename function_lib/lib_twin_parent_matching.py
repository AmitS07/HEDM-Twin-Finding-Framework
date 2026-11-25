'''
Date: 17th May 2024
Author: Amit Singh, ACDM lab
Description: This library-script does find the grains-matching for data set of each laod step

'''

##########################################################################
#%% 0. Importing Libaries
##########################################################################
#0. Importing libraries from in-bulid python
import pandas as pd
import numpy as np
import json
import os
import glob
import time
from math import pi, cos, sin, acos, sqrt, tan
import random
from scipy.spatial.transform import Rotation as ROT
from collections import Counter

#0.2 Importing written- libraries-script
from lib_orientations_symmetries import *
from lib_functions_analysis import *


##########################################################################
#%% 1. Function for grains matching across load steps
##########################################################################
'''
def twin_finding(df1,df2,tol_misori,wt_misori):
    # df1 = Dataframe of layer1, df2 =  Dataframe of layer2
    # tol_euclid = euclidean distance tolerance, tol_misori = misorientation tolerance
    # wt_misori = weightage to the misorientation 

    ############# The function start from here ############
    # reading the rotation matrix column name
    rot_names = df1.columns[1:10].to_list() # slecting out the commom grains

    ## The tolerances in orientation 
    misori_tol = tol_misori
    ## Weightage for confidences in "orientation" and "position"
    ws = [wt_misori, 1-wt_misori]

    ############ Calculating the twins bw df1  ############
    ####################################################################################################### 

    matched_grains_l12 = []
    # for l1_idx, l1_vals in df1[df1['Z']<-280].iterrows(): #This to run the test very fast on selective grains near to boundary bw two layers
    for l1_idx, l1_vals in df1.iterrows(): # This is to run raw the grains bw two layers
        # Old Grains's IDs name
        old_ID1 = int(df1.loc[l1_idx]['GrainID'])
        # Factor1: Orientations
        rot1 = rot2mat(l1_vals[rot_names].to_list())
        q1 = rot2quat_scipy(rot1)
        # Factor2: Radius
        r1 = df1['GrainRadius'].loc[l1_idx]
        # Factor3: Sptial coordinates
        cord1 = df1[['X', 'Y', 'Z']].loc[l1_idx]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #### The tolerances in position x,y,z i.e cord  = coordinate
        x_tol = y_tol = z_tol = r1
        cord_tol = [x_tol, y_tol, z_tol]
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # #%%%%%%%%%%%%%%%%%%%%%% comparing Layer1-2 #%%%%%%%%%%%%%%%%%%%%%% 
        # for l2_idx, l2_vals in df2[df2['Z']>-320].iterrows(): #This to run the test very fast on selective grains near to boundary bw two layers
        for l2_idx, l2_vals in df2.iterrows(): #This is to run raw the grains bw two layers
            # Old Grains's IDs
            old_ID2 = int(df2.loc[l2_idx]['GrainID'])
            # Factor1: Orientations
            rot2 = rot2mat(l2_vals[rot_names].to_list())
            q2 = rot2quat_scipy(rot2)
            # Factor2: Radius
            r2 = df2['GrainRadius'].loc[l2_idx]
            # Factor3: Sptial coordinates
            cord2 = df2[['X', 'Y', 'Z']].loc[l2_idx]
            
            ##%%%%%%%%%%%%%%%%%%%%%%
            # Estimating misori_angle/ and Euclid_dist
            eucid_dist_l12 = euclidean_dist(cord2,cord1) #eucid_dist_l12 = np.linalg.norm(np.array(cord2)-np.array(cord1))
            trans_euclid = euclidean_dist([cord2[0],cord2[1]],[cord1[0],cord1[1]])#trans_euclid = np.linalg.norm(np.array([cord2[0],cord2[1]])-np.array([cord1[0],cord1[1]]))
            if eucid_dist_l12<=r1 and trans_euclid<=r1: # euclidean_dist was fliterout based on "radius" a grain in load-1
                misori_angle_l12 = mis_ori(q1, q2)
                if misori_angle_l12<=misori_tol: 
                    rel_rad = abs(r1-r2)/r1 
                    rad_tol =  r1*0.01
                    if rel_rad<=rad_tol: # This to confirm that the grain shold be relatively in same size
                        # Estimating Confidence: misori_CI and centroid_CI
                        misori_CI_l12 = confidence(misori_angle_l12,misori_tol)
                        cord_CI_l12 = centroid_confidence(cord1[0:2],cord2[0:2],cord_tol, domain="rectangular")  
                        CIs_l12 = [misori_CI_l12,cord_CI_l12] # combining and weighing both CIs
                        comul_CI_l12 = cumulative_confidence(CIs_l12, ws)
                        # The list of "matched_grains"
                        matched_grains_l12.append([old_ID1,old_ID2,l1_idx, l2_idx, round(misori_angle_l12,4), round(eucid_dist_l12,4),round(trans_euclid,4), 
                                            round(misori_CI_l12,4), round(cord_CI_l12,4), round(comul_CI_l12,4)])
                
    ### saving the matched_grains
    matched_grains_l12 = pd.DataFrame(matched_grains_l12) # This is to convert list to data_frame
    #*****Note: This below Coulmn Name will work only when there should be at least data set, if there is 0 data then you have to comment it and uncommet from the top ************
    matched_grains_l12.columns = ['OldGrainsID_L1','OldGrainsID_L2','NewGrains_ID_L1', 'NewGrains_ID_L2', 'Misorientation_angle', 'Euclidean_Dist','Trans_euclid', 'Misori_CI', 'Centroid_CI', 'Comulative_CI']
    df12_matched_grains = matched_grains_l12.sort_values(by = 'Misorientation_angle') # just to sort-out i.e arranging them based on accending order
    
    return df12_matched_grains
''' 
##########################################################################
#%% 2. Function for twin_reading and grains matching
##########################################################################
#2.1 Read Dictionary saved as 'csv'
def read_csv_dictionary(csv_file_path):
    # Required libraries
    import csv
    import ast
    # defining 'dictionary'
    dict= {}
    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for rows in reader:
            key = int(rows[0]) # The first column is the key
            values = [ast.literal_eval(v) for v in rows[1:]] # The remaining columns are the values
            dict[key] = values
    return dict

#2.2 To get the dictionary Values based on Key
def get_dictionary_values(dictionary, key):
    if key in dictionary:
        values = dictionary[key][0] # Extract the list of lists
        value1 = np.array(values[0],dtype=float)
        value2 = np.array(values[1],dtype=float)
        return value1, value2
    else: 
        return None, None
#%%2.3 data frame to list for the specific column
# def df2list_specific_columns(ld_s,icrmt,ld_f,tw_id):
#     # This function convert the  dataframe to list, 
#     # but, this is for specific column name ie.e "Load_0", "Load_25" ....
#     #----------------------------------------------------------------
#     columns_name = []
#     while ld_s<=ld_f: # ld = load
#         columns = f"Load_{ld_s}" # name of the column
#         columns_name.append(columns)
#         ld_s += icrmt
#     #----------------------------------------------------------------
#     row_list = tw_id[columns_name].values.flatten().tolist()
#     return row_list
def df2list_specific_columns(load_range,ids):
    # This function convert the  dataframe to list, 
    # but, this is for specific column name ie.e "Load_0", "Load_25" ....
    #----------------------------------------------------------------
    columns_name = []
    for ld_c in load_range:
        columns = f"Load_{ld_c}" # name of the column
        columns_name.append(columns)
    #----------------------------------------------------------------
    row_list = ids[columns_name].values.flatten().tolist()
    return row_list

#%% 2.4 Function to short parent-grains-IDs
def sorting_pgIDs_loadwise(load_range, ids_loadwise):
    '''
    This function does sorting of parnet-grains ID-s from the saved dictionary of the parent-twin pair
    Note: This is very specific function if dictionary are saved by key "LS_xxx_PG" where xxx is the load
    Output: The output will be also be dictionary with key "LS_yy", where yy stand for load-step number
    '''
    sorted_ids_ls = {} # This dictionary contains all the parent-grains for load-step:0 in increment of load-steps from 0-300, corresponding to all the unique twinned parnet-grains
    for ls in range(0,len(load_range)):
        ids_starting = []
        for load_c in load_range:
            # Reading number of keys for each dictionary to find the number of twins i.e corresponding parent-grains
            #%% This can also be done like above just by reading the IDs_pg from the get dictionary value
            #%***********************************************************************
            n_ids = len([k for k in ids_loadwise.keys() if k.startswith(f"load_{load_c}_pg")])
            for i in range(1,n_ids+1): # looping to get the parent_grain from load-step:0 
                    ids_starting.append(ids_loadwise[f"load_{load_c}_pg_{i}"][ls]) # this [ls] ensure, parent-grains from load:0
        sorted_ids_ls[f"ls_{ls}"] = ids_starting #unique() does select out only one number out of repeated Ids in accending order 
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Note*: After using unique this data is not in same order as it saved  in the load-wise merged grains, so avoid using "unique()"  in above line. 
    return sorted_ids_ls
#%% Function to short twin-IDs
def sorting_twIDs_loadwise(idx_load_twins,load_range_twin,tw_ids_LS):
    '''
    Input:: 'load_range_i': load range from start i.e. load-0 to final load, bcoz given load_range are not from load-0
            'twin_load': the load from which one want to analyze the twinning
            'tw_ids_Ls': list of twin IDs load-step wise
    Description: This function does sorting of parnet-grains ID-s from the saved dictionary of the parent-twin pair
    Note: This is very specific function if dictionary are saved by key "LS_xxx_Twin" where 'xxx' is the load
    Output: The output will be also be dictionary with key "LS_ls", where 'ls' stand for load-step number
    '''
    # based on to choose the twin form load-steps or load
    twIDs_sorted_ls = {} # This dictionary contains all the parent-grains for load-step:0 in increment of load-steps from 0-300, corresponding to all the unique twinned parnet-grains
    ## Select load from which one want to analyze the twinnning
    for ls in idx_load_twins:
        ids_starting = []
        for load_c in load_range_twin:
            # Reading number of keys for each dictionary to find the number of twins i.e corresponding parent-grains
            #%% This can also be done like above just by reading the IDs_pg from the get dictionary value
            #%***********************************************************************
            n_ids = len([k for k in tw_ids_LS.keys() if k.startswith(f"load_{load_c}_tw")])
            for i in range(1,n_ids+1): # looping to get the parent_grain from load-step:0 
                    ids_starting.append(tw_ids_LS[f"load_{load_c}_tw_{i}"][ls-1]) # this [ls] ensure, parent-grains from load:0
        twIDs_sorted_ls[f"ls_{ls}"] = ids_starting #unique() does select out only one number out of repeated Ids in accending order 
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Note*: After using unique this data is not in same order as it saved  in the load-wise merged grains, so avoid using "unique()"  in above line. 
    return twIDs_sorted_ls



