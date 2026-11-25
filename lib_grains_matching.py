'''
Date: Saturday Sept. 26 2023
Author: Amit, ACDM Lab
Description: This library contaions all the function for all analysis related to grains matching,
              across layer and load wise, for in-house develop code

'''
## This code contaions all the function for all analysis related to texture and grains matching, twinning
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

# importing manually defined functions 
from lib_orientations_symmetries import *
from lib_functions_analysis import *

##########################################################################
#%% 5. Function for grains matching across layer
##########################################################################
## 5.1
def grains_matching_layerwise(df1,df2,tol_euclid,tol_misori,wt_misori):
    # This function, will findout the matched grains between given two layers
    # df1 = Dataframe of layer1, df2 =  Dataframe of layer2
    # tol_euclid = euclidean distance tolerance, tol_misori = misorientation tolerance
    # wt_misori = weightage to the misorientation 

    ############# The function start from here ############
    # reading the rotation matrix column name
    rot_names = df1.columns[1:10].to_list() # slecting out the commom grains
    ## The tolerances in position x,y,z i.e cord  = coordinate
    x_tol = y_tol = z_tol = euclid_tol = tol_euclid
    cord_tol = [x_tol, y_tol, z_tol]
    ## The tolerances in orientation 
    misori_tol = tol_misori
    ## Weightage for confidences in "orientation" and "position"
    ws = [wt_misori, 1-wt_misori]

    ############ Calculating the matched grains between layer1 and layer2 = matched_grains_l12 ############
    ####################################################################################################### 

    matched_grains_l12 = []
    # for l1_idx, l1_vals in df1[df1['Z']<-280].iterrows(): #This to run the test very fast on selective grains near to boundary bw two layers
    for l1_idx, l1_vals in df1.iterrows(): # This is to run raw the grains bw two layers
        # Old Grains's IDs name
        old_ID1 = int(df1.loc[l1_idx]['GrainID'])
        # Factor1: Orientations
        rot1 = rot2mat(l1_vals[rot_names].to_list())
        q1 = rot2quat_scipy(rot1)
        # # Factor2: Radius
        # rad1 = df1['GrainRadius'].loc[l2_idx]
        # Factor3: Sptial coordinates
        cord1 = df1[['X', 'Y', 'Z']].loc[l1_idx]

        # #%%%%%%%%%%%%%%%%%%%%%% comparing Layer1-2 #%%%%%%%%%%%%%%%%%%%%%% 
        # for l2_idx, l2_vals in df2[df2['Z']>-320].iterrows(): #This to run the test very fast on selective grains near to boundary bw two layers
        for l2_idx, l2_vals in df2.iterrows(): #This is to run raw the grains bw two layers
            # Old Grains's IDs
            old_ID2 = int(df2.loc[l2_idx]['GrainID'])
            # Factor1: Orientations
            rot2 = rot2mat(l2_vals[rot_names].to_list())
            q2 = rot2quat_scipy(rot2)
            # # Factor2: Radius
            # rad2 = df2['GrainRadius'].loc[l2_idx]
            # Factor3: Sptial coordinates
            cord2 = df2[['X', 'Y', 'Z']].loc[l2_idx]
            
            ##%%%%%%%%%%%%%%%%%%%%%%
            # Estimating misori_angle/ and Euclid_dist
            misori_angle_l12 = mis_ori(q1, q2)
            eucid_dist_l12 = euclidean_dist(cord2,cord1) #eucid_dist_l12 = np.linalg.norm(np.array(cord2)-np.array(cord1))
            if misori_angle_l12<=misori_tol and eucid_dist_l12<=euclid_tol:
                trans_euclid = euclidean_dist([cord2[0],cord2[1]],[cord1[0],cord1[1]]) #trans_euclid = np.linalg.norm(np.array([cord2[0],cord2[1]])-np.array([cord1[0],cord1[1]]))
                if trans_euclid<=euclid_tol:                      
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

##########################################################################
#%% 6. Function for merging two layers
##########################################################################
## 6.1 remove extra similar grains (based on volume weightage) from the same dataframe
def remove_same_extra_grain(df, grainID1, grainID2):
    #Input: df= Dataframe from which the grain/s has to be removed, here either df1 or df2 if matched grains from layer12 and layer23, respeectively 
    #Output: df_new = new reduced dataframe fater cutting/removing same extra grain/s based on vvolume weighatge
    rad1 = df['GrainRadius'].loc[grainID1]
    rad2 = df['GrainRadius'].loc[grainID2]
    wt_v = volume_weightage(rad1,rad2)
    w1,w2 = wt_v[0],wt_v[1]
    if w1>w2:
        # Matched grains filter based on 
        df_reduced = df[df.index.isin([grainID2])]
        # Drop the matched grains
        df_new = df.drop(df_reduced.index)
    else:
        df_reduced = df[df.index.isin([grainID1])]
        # Drop the matched grains
        df_new = df.drop(df_reduced.index)
    return df_new,w1,w2

## 6.2 checking and removing extra similar grain/s based on volume weighatge from the raw data frame of either layer1 or layer2
def check_and_remove_double_match_grains(df_matched_grains,df_raw):
    ## input: (1) df_matched_grains = Dataframe of merged grains b/w two-layer (either 1-2layer or 2-3layer)
    ##        (2) df_raw = raw data of either layer1 or layer 2 if df_matched_grains is 1-2 or 2-3 respectively
    ## output = reduced data_raw after cutting double matched grains either from df1 or df2 
    matched_grainsIDs_L2 = df_matched_grains['NewGrains_ID_L2'] # 'df_matched_grains_sort' is a data of matched grains with given index name
    double_matched_grainIDs_L2 = np.unique(matched_grainsIDs_L2)
    if len(matched_grainsIDs_L2) != len(double_matched_grainIDs_L2):
        print("There are double matched grains b/w two layer")
        ### If true, then do the further analysis ###
        ## Estimating the double matched_grains i.e non_unique-grains 
        non_unique_grains = non_unique_elements(matched_grainsIDs_L2) ## "non_unique_elements" is a user define function
        ## Selecting out elements/grains from "NewGrains_ID_L1" (a column) based on elements from "NewGrains_ID_L2" (given other column)
        for elements in non_unique_grains:
            selected_elements = df_matched_grains[df_matched_grains['NewGrains_ID_L2'] == elements]['NewGrains_ID_L1'].tolist()
        
        ## Removing extra grain from the raw_data based on selected_elements
        i = 0 # initializing the count
        while i <= len(selected_elements)/2: 
            df_cleaned,w_id1,w_id2 = remove_same_extra_grain(df_raw,selected_elements[i],selected_elements[i+1])
            df_raw = df_cleaned
            ## Removing extra grains from the "df_mathced_grains"
            if w_id1>w_id2:
                df_matched_new = df_matched_grains.drop(df_matched_grains[df_matched_grains['NewGrains_ID_L1'] == selected_elements[i+1]].index)
            else:
                df_matched_new = df_matched_grains.drop(df_matched_grains[df_matched_grains['NewGrains_ID_L1'] == selected_elements[i]].index)
            ##################
            # looping for next set
            i +=2
        df_modified = df_raw
    else:
        ## if this, good to go, do nothing
        print("There are not any double matched grains b/w two layer")
        ## just chnage the file names
        df_modified = df_raw
        df_matched_new = df_matched_grains
        
    return df_modified,df_matched_new


## 6.3 merging layer function
def merged_grains_two_layers(df1,df2,matched_grains_l12):
    # This function gives the total merged grains after considering matched grains pairs common bw two layes
    ## Required inputs: df1 = raw dataframe of layer1, df2 = raw dataframe of layer2, matched_grains_l12 = list of matched grains bw two layers
    # Just to note, df1,df2 are the actual all grain raw data of layer1 and layer 2, i.e. df1_all and df2_all
    l1_matched_grains,l2_matched_grains = matched_grains_l12.iloc[:,2], matched_grains_l12.iloc[:,3]
    ## Code start from here
    counter_1  = 0
    counter_2  = 0
    for l1_id1, l2_id2 in zip(l1_matched_grains,l2_matched_grains):
        rad1 = df1['GrainRadius'].loc[l1_id1]
        rad2 = df2['GrainRadius'].loc[l2_id2]
        wt_v = volume_weightage(rad1,rad2)
        w1,w2 = wt_v[0],wt_v[1]
    #### Note: Here, which one has more wt., that will we considered not "averaged one"
    ### Average orientation analysis will be carried out later
        if w1>w2:
            # W1>W2, means "layer1" has more weightage than "layer2"
            counter_1 += 1
            if counter_1 == 1:
                # Matched grains filter and droping grains
                df2_matched = df2[df2.index.isin([l2_id2])]
                # Drop the matched grains
                df2_new = df2.drop(df2_matched.index)
            else:
                # Matched grains filter based on 
                df2_matched = df2_new[df2_new.index.isin([l2_id2])]
                # Drop the matched grains
                df2_new = df2_new.drop(df2_matched.index)
            # make no change in the "df1" data
            if counter_2 == 0:
                df1_new = df1
            else:
                df1_new = df1_new
            
        else:
            # W2>W1, means "layer2" has more weightage than "layer1"
            counter_2 += 1
            if counter_2 == 1:
                # Matched grains filter based on 
                df1_matched = df1[df1.index.isin([l1_id1])]
                # Drop the matched 1grains
                df1_new = df1.drop(df1_matched.index)
            else:
                # Filtering rows to be removed
                df1_matched = df1_new[df1_new.index.isin([l1_id1])]
                # Drop the matched grains
                df1_new = df1_new.drop(df1_matched.index)
            # make no change in the "df2" data
            if counter_1 ==0:
                df2_new = df2
            else:
                df2_new = df2_new
    ## Merging the unique grains
    # df12 = pd.concat([df1_new, df2_new])
    return df1_new,df2_new


##########################################################################
#%% 5. Function for grains matching across load steps
##########################################################################
## 5.1
def grains_matching_loadwise(df1,df2,tol_radius,tol_misori,wt_misori):
    # This function, will findout the matched grains between given two load steps: "in-house" script
    # df1 = Dataframe of layer1, df2 =  Dataframe of layer2
    # tol_euclid = euclidean distance tolerance, tol_misori = misorientation tolerance
    # wt_misori = weightage to the misorientation 
    ############# The function start from here ############
    # reading the rotation matrix column name
    # rot_names = df1.columns[1:10].to_list() # slecting out the commom grains
    ## The tolerances in orientation 
    misori_tol = tol_misori
    rad_tol  = tol_radius
    ## Weightage for confidences in "orientation" and "position"
    ws = [wt_misori, 1-wt_misori]

    ############ Calculating the matched grains between layer1 and layer2 = matched_grains_l12 ############
    ####################################################################################################### 

    matched_grains_l12 = []
    # for l1_idx, l1_vals in df1[df1['Z']<-280].iterrows(): #This to run the test very fast on selective grains near to boundary bw two layers
    for l1_idx, l1_vals in df1.iterrows(): # This is to run raw the grains bw two layers
        # Old Grains's IDs name
        old_ID1 = int(df1.loc[l1_idx]['GrainID'])
        # Factor1: Orientations
        rot1 = rot2mat(l1_vals["O11":"O33"].to_list())
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
            rot2 = rot2mat(l2_vals["O11":"O33"].to_list())
            q2 = rot2quat_scipy(rot2)
            # Factor2: Radius
            r2 = df2['GrainRadius'].loc[l2_idx]
            # Factor3: Sptial coordinates
            cord2 = df2[['X', 'Y', 'Z']].loc[l2_idx]
            ##%%%%%%%%%%%%%%%%%%%%%%
            # Estimating misori_angle/ and Euclid_dist
            eucid_dist_l12 = euclidean_dist(cord2,cord1) #eucid_dist_l12 = np.linalg.norm(np.array(cord2)-np.array(cord1))
            trans_euclid = euclidean_dist([cord2[0],cord2[1]],[cord1[0],cord1[1]]) #trans_euclid = np.linalg.norm(np.array([cord2[0],cord2[1]])-np.array([cord1[0],cord1[1]]))
            if eucid_dist_l12<=r1+rad_tol and trans_euclid<=r1+rad_tol: # euclidean_dist was fliterout based on "radius" a grain in load-1
                misori_angle_l12 = mis_ori(q1, q2)
                if misori_angle_l12<=misori_tol: 
                    rel_rad = abs(r1-r2)/r1  ## Based on Prof.Hamid Abdolvand work
                    rel_rad_tol =  r1*0.1
                    if rel_rad<=rel_rad_tol: # This to confirm that the grain shold be relatively in same size
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
#%%-------------------------------------------------------------------------------------------------------------
## 5.2, Misorienation and Euclidean dist calculation between two-consecutive load-steps
def calc_misori_euclideanDist_bw_twoLoad(df1, df2):
    ## This version save the rows and columns as the grains_ids1 and grainsID2.
    M, N = len(df1), len(df2)
    # Prepare empty object array for DataFrame cells to hold "lists", as reuslt the output array will be string
    grainID1 = [str(idx) for idx in df1.index]
    grainID2 = [str(idx) for idx in df2.index]
    grains_matrix_l12 = np.empty((M, N), dtype=object)
    ## Iterating over each grains for df1 and df2
    for i, (l1_idx, l1_vals) in enumerate(df1.iterrows()):
        rot1 = rot2mat(l1_vals["O11":"O33"].to_list())
        q1 = rot2quat_scipy(rot1)
        cord1 = df1[['X', 'Y', 'Z']].loc[l1_idx]
        #********************************************************
        for j, (l2_idx, l2_vals) in enumerate(df2.iterrows()):
            rot2 = rot2mat(l2_vals["O11":"O33"].to_list())
            q2 = rot2quat_scipy(rot2)
            cord2 = df2[['X', 'Y', 'Z']].loc[l2_idx]
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ## Calculations for Euclidean-dist, trans_eucliden_dist, misori
            eucid_dist = euclidean_dist(cord2, cord1)
            misori_l12 = mis_ori(q1, q2)
            # Store list of five values in each cell
            grains_matrix_l12[i, j] = [round(misori_l12,5), round(eucid_dist,5)]
    # Create DataFrame
    df_grains_l12 = pd.DataFrame(data = grains_matrix_l12, index = grainID1, columns=grainID2)

    return df_grains_l12
#%%----------------------------------------------------------------------------------------------------------
##5.3: Similar to MIDAS approach of calculation for the "misori" and "eucllid"
def map_misori_euclideanDist_bw_twoLoad(df1_all, df2_all,w_misori):
    M, N = len(df1_all), len(df2_all) ## Defining the dimension of the array based on the maximum 
    grains_matrix_l12 = np.zeros((max(M,N), 4))
    grainID1 = df1_all.index.tolist()
    grainID2 = df2_all.index.tolist()

    count = 0
    while grainID1 and grainID2:
        gID1 = grainID1[0]
        rot1 = rot2mat(df1_all.loc[gID1]["O11":"O33"])
        q1 = rot2quat_scipy(rot1)
        cord1 = df1_all.loc[gID1][['X', 'Y', 'Z']]

        misori_list, euclid_list = [], []
        for gID2 in grainID2:
            rot2 = rot2mat(df2_all.loc[gID2]["O11":"O33"])
            q2 = rot2quat_scipy(rot2)
            cord2 = df2_all.loc[gID2][['X', 'Y', 'Z']]
            eucid_dist = euclidean_dist(cord2, cord1)
            misori_l12 = mis_ori(q1, q2)
            misori_list.append(misori_l12)
            euclid_list.append(eucid_dist)
        ## To acccount for both misri and euclid  in filtering the grains with weight
        misori_arr = np.array(misori_list)
        euclid_arr = np.array(euclid_list)
        misori_norm = (misori_arr - np.min(misori_arr)) / (np.ptp(misori_arr) + 1e-9)
        euclid_norm = (euclid_arr - np.min(euclid_arr)) / (np.ptp(euclid_arr) + 1e-9)
        combined = w_misori*misori_norm + (1-w_misori)*euclid_norm  # Adjust weights as preferred
        min_index = np.argmin(combined)

        # min_index = np.argmin(misori_list)  # or euclid_list as needed
        min_gID2 = grainID2[min_index]
        min_misori = misori_list[min_index]
        min_euclid = euclid_list[min_index]

        grains_matrix_l12[count, 0] = gID1
        grains_matrix_l12[count, 1] = min_gID2
        grains_matrix_l12[count, 2] = round(min_misori,5)
        grains_matrix_l12[count, 3] = round(min_euclid,5)
        count += 1

        # After matching, drop matched grain IDs from both
        df1_all = df1_all.drop(gID1)
        df2_all = df2_all.drop(min_gID2)
        grainID1 = df1_all.index.tolist()
        grainID2 = df2_all.index.tolist()

    df_grains_l12 = pd.DataFrame(grains_matrix_l12[:count])
    df_grains_l12.columns = ["OrigIDState1", "OrigIDState2", "minAngle", "EuclideanDistt"]
    return df_grains_l12


#########################################################################################################################
#%% 6. Function to find twin-parent pair at each load for all loads
#########################################################################################################################
def twin_parent_pair_finding_at_load(scale_factor, load_c, df_load, ptw_IDs, pg_IDs, avg_rad_factor, approach, pg_rad_tol, tw_misori_angle, twin_tol, quat_th, consider):
    '''
    Modified (02/11/2026): To account for axis of misorientation for selecting parent and twin
    '''
    #%% Defining variables to store the data 
    twin_pg_IDs = [] #%% To Store Found twin-IDs as list
    scale_factor[f"Load-{load_c}"] = []
    id_pg_min_euclid = {} # dictionary of parent_grain's id with minimum euclidean dist
    #%%+++++++++++++++++++++++++++++++++++++++ Iteratively analyzing for each grain +++++++++++++++++++++++++++++++++++
    for id_tw in ptw_IDs: # Looping over each ID as potential-twin
        rot_tw = df_load.loc[id_tw]["O11":"O33"]
        quat_tw = rot2quat_scipy(rot2mat(rot_tw.to_list()))
        cord_tw = df_load.loc[id_tw]["X":"Z"]
        rad_tw = df_load.loc[id_tw]["GrainRadius"]*avg_rad_factor
        for id_pg in pg_IDs: # Looping over each ID  as  parent-grain 
            rot_pg = df_load.loc[id_pg]["O11":"O33"]
            quat_pg = rot2quat_scipy(rot2mat(rot_pg.to_list()))
            cord_pg = df_load.loc[id_pg]["X":"Z"]
            rad_pg = df_load.loc[id_pg]["GrainRadius"]*avg_rad_factor
            #%%++++++++++++++++++++++++++++++++++ Incorporating twins criteria ++++++++++++++++++++++++++++++++++
            misori_twin, _,_,_ = mis_ori(quat_pg, quat_tw, sym_operators=HEXSYM, degree=True)
            euclid_twin = np.linalg.norm(np.array(cord_tw)-np.array(cord_pg))
            if approach == "wr": #Radius corrected by the size of measured volume of all grains and domain volume     
                if rad_tw<=rad_pg: # Twin's size should be less than parent grain's size, 
                    # scale_factor[f"Load-{load_c}"].append(euclid_twin/rad_pg) # This is just to test the relative size between for all rad_tw<rad_pg
                    twin_euclid_tol = rad_pg + rad_tw + pg_rad_tol # Twin's euclidean dist tolerance based on parent grain radius
                else:
                    pass
            else:
                twin_euclid_tol = rad_pg + rad_tw + pg_rad_tol # Ref: https://doi.org/10.1038/s43246-020-00105-y
            #%%++++++++++++++++++++++++++++++++ Filtering twins +++++++++++++++++++++++++++++++++
            if abs(misori_twin - tw_misori_angle)<=twin_tol and euclid_twin<=twin_euclid_tol:
                scale_factor[f"Load-{load_c}"].append(euclid_twin/rad_pg) # This is just to test the relative size between distnace between eucliden dist wrt twin
                _, _, misori_deviation = misori_deviation_check(quat_pg, quat_tw, quat_th, sym=HEXSYM, degree=True)
                if misori_deviation["delta_theta"]<=twin_tol and misori_deviation["delta_alpha"]<=twin_tol*2:
                #%%----------------------------All or Unique Parent-Grains for a twin-----------------------------------
                    if consider=="all": #%%To consider all possible PARENT-GRAIN for a twin
                        twin_pg_IDs.append([id_tw, id_pg]) #%% if you want to estmimate the unique twins. Note: dictionary left element is "twin_ids" and right is "parent_ids"
                    else: #%%To find "UNIQUE" PARENT-GRAIN for a twin
                        if id_tw in id_pg_min_euclid:
                            # Compare the current euclid_twin with the stored one
                            if euclid_twin < id_pg_min_euclid[id_tw]['euclid_twin']:
                                # Update the dictionary with the new lower distance and corresponding id_pg
                                id_pg_min_euclid[id_tw] = {'euclid_twin': euclid_twin, 'id_pg_unique': id_pg}
                        else: # Add new entry for id_tw
                            id_pg_min_euclid[id_tw] = {'euclid_twin': euclid_twin, 'id_pg_unique': id_pg}
    #%%--------------------------------------------------------------------------
    #%% After the loop, store the pairs in twin_pg_IDs, for for Unique grains
    if consider == "unique":
        for id_tw, id_pg in id_pg_min_euclid.items():
            twin_pg_IDs.append([id_tw, id_pg['id_pg_unique']]) # Note: dictionary left element is "twin_ids" and right is "parent_ids"
    
    return twin_pg_IDs, scale_factor