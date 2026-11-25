'''
Author: Amit Singh
Created: 22th March 2024

This is a function library for grains matching acrosss load steps using MIDAS (APS software)

'''

# importing libaries
# from in-build python  
import os
import glob
import pandas as pd
import numpy as np

# Importing library manual defined
from lib_functions_analysis import *
from lib_mathematics import *

## Defing function 

def parquet_to_csv_midas(pth, sam_n, load_n):
    '''
    This function is to convert "parquet" file of raw data to "csv" format exactly in same format as MIDAS requires
    Note: This is a very specific conversion particular for MIDAS data
    Input: path = path of the directry
           sam_n  =  Smaple number, as it assign during the experiment at APS
           load_n =  Load number i.e. load which were used during the in-situ synchrotron test
    Output: It just save the data-farme as per MIDAS requirment in "csv" format, these is not any output
    '''
    df = pd.read_parquet(f'{pth}/sample_{sam_n}/parquet_files/sam{sam_n}_{load_n}N_layered_grains.parquet')
    print(f"#Grains in sample {sam_n} Load {load_n} :", len(df))
    df = df.reset_index()
    del df["GrainID"] # del df[df.columns[1]]
    df['new_GrainID'] = df['new_GrainID'].astype(str).str[1:5] + df['new_GrainID'].astype(str).str[6:]
    df.rename(columns={'new_GrainID': '%new_GrainID'}, inplace=True) ## To add "%" in first place of the 1st column of 1st row
    #%% Saving the data, after replacing the "%new_GrainID" with "GrainID"
    df.to_csv(f'{pth}/sample_{sam_n}/csv_files/merged_layer_data/sam{sam_n}_{load_n}N_layer_grains.csv',sep='\t',index=False)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    ## Adding headers on merged_data_layer
    #% Reading back the just above saved merged data
    with open(f'{pth}/sample_{sam_n}/csv_files/merged_layer_data/sam{sam_n}_{load_n}N_layer_grains.csv','r') as merged_data:
        merged_layer_data = merged_data.readlines()
    #% Reading headers from textfiles 
    with open(f'{pth}/sample_{sam_n}/txt_files/sam{sam_n}_{load_n}N_Layer1_headers.txt', 'r') as headers:
        firstline = headers.readline() #%% for all lines i.e 8 lines headers.readlines()
    #%% Appending both to one file
    texts =[firstline] + merged_layer_data #%% for all lines i,e 8 lines texts = firstline +merged_layer_data
    with open(f'{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_n}N_grains.csv', 'w') as writer:
        writer.writelines(texts)
#%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def MIDAS_grains_match(z_offset, beamThickness, twinAngle, OutFile, InputFile1, InputFile2):
    '''
    It run the MIDAS on you local machine for two data sets
    Input: z_offset = This should be given to tolerate the "beam movement"
           beamThickness = it is given in data, is generally height of the beam
           twinAngle = It's reference angle for twin eg 85 (degree) if 85, then misorientation calculated with respect to this angle
           path = path of the directry
           sam_n  =  Smaple number, as it assign during the experiment at APS
           load_n =  Load number i.e. load which were used during the in-situ synchrotron test
           incrmt = The difference between the twp data sets, lets in load 
    Output: 
            It only run the MIDAS with given data and save the matched grains list 
    '''
    SGNr = 194 #space group for orientation matching
    offset_x = 0 # X in MIDAS is along the x-ray beam
    offset_y = 0 # Y is horizontal in a right handed coordinate system
    offset_z = z_offset
    matchMode = 2 #Match according to both orientation and position using supplied weights
    beamThickness2 = beamThickness1 = beamThickness # Beam height or total area
    removeDuplicates = 1 #will remove any matched grains from database while matching. 
    sizeFilter = 0 #0 will not make filter based on grain size
    weights_deg = 0.5 #provide relative weights to normalize orientation and position resolution.  i.e misori/weights_deg
    weights_mic = 50 # provide relative weights to normalize position resolution.  i.e COM(position)/weights_mic
    # Based on thes weighatage: selecting criteria calculated. These two should be fixed and could be considered as an uppar bound
    referenceMisorientation = twinAngle # to calculate misorientation with respect to twin's ori = 85

    # MIDAS directory source link
    __scr_base__ = '~/opt/MIDAS/FF_HEDM/bin/MatchGrains'
  
    # Grains mathcing a sper MIDAS format
    scr = f'{__scr_base__} '
    scr += f'{OutFile} '
    scr += f'{InputFile1[0]} {InputFile2[0]} '
    scr += f'{SGNr} '
    scr += f'{offset_x:.3f} {offset_y:.3f} {offset_z:.3f} '
    scr += f'{matchMode} '
    scr += f'{beamThickness1} {beamThickness2} '
    scr += f'{removeDuplicates} '
    scr += f'{sizeFilter} '
    scr += f'{weights_deg} {weights_mic} '
    scr += f'{referenceMisorientation} '
    
    print(scr)
    print(20*'***')
    # Running MIDAS
    os.system(scr)


def filter_matched_grainsIDs(df_matched, tol_misori, tol_euclid):
    euclid = df_matched['EuclideanDistt']
    misori = df_matched['minAngle']
    ## listing out mathced grains' IDs
    matched_grainsID_load1 = []
    matched_grainsID_load2 = []
    for i,(m,e) in enumerate(zip(misori,euclid)):
        if m<=tol_misori and e<=tol_euclid:
            load1_Ids = df_matched['OrigIDState1'].iloc[i]
            matched_grainsID_load1.append(load1_Ids)
            load2_Ids = df_matched['OrigIDState2'].iloc[i]
            matched_grainsID_load2.append(load2_Ids)
    return matched_grainsID_load1,matched_grainsID_load2


###########################################################################################
########################### Part-2: Selecting out Matched grains ##########################
###########################################################################################

def common_matched_grains(match_grainsID_L123, match_grainsID_L2_n, match_grainsID_L3):
    '''
    Input: match_grainsID_L123 = list of matched grains b/w two consective load-steps
           match_grainsID_L2_n = list of matched grains Ids form (n+1)th load step
           match_grainsID_L3 = list of matched grains Ids form nth load step
    Output: Gives common grains between the given two consective load steps, ...
            based on given matched_grains list
    '''
    df_matched_L12 = pd.DataFrame(match_grainsID_L123) 
    df_L2_n = pd.DataFrame(match_grainsID_L2_n)
    df_L3 = pd.DataFrame(match_grainsID_L3)

    common_grains_L3 = []
    for val in df_matched_L12.iloc[:,0]:
        idx = df_L2_n.index[df_L2_n.iloc[:,0] == val].values
        final_grains_L3 = df_L3.iloc[idx,0]
        common_grains_L3.extend(final_grains_L3)
    return common_grains_L3

def matched_unmatched_grains_loadstep(df_load1, matched_grains_id1):
    ''' 
    This function find out the matched and unmatched grains beetween two data sets, by giving list to matched grains IDs
    Input: df_load1 = dataframe of a load1
           df_load2 = dataframe of load2
           matched_grains_id1 = Grains ID of matched grains of load1
           matched_grains_id2 = Grains ID of matched grains of load2
    Outut: dataframes of matched/unmatched grains of load1 and load2 respectively ...
           df1_matched, df2_matched, df1_unmatched, df2_unmatched
    '''
    ## Distingusing matched and unmatched grains
    df1_matched = []
    counter = 1
    for id1 in matched_grains_id1:
        if counter == 1:
            # Finding Matched grain from load(i) data set
            df_matched1 = df_load1[df_load1.index.isin([id1])]
            df1_matched.append(df_matched1)
            # Drop the matched grains and making unmathced grains for load(i) data
            df1_unmatched = df_load1.drop(df_matched1.index)
            counter +=1
        else:
            # Matched grains filter based on 
            df_matched1 = df1_unmatched[df1_unmatched.index.isin([id1])]
            df1_matched.append(df_matched1)
            # Drop the matched grains
            df1_unmatched = df1_unmatched.drop(df_matched1.index)
    ## To change list to dataframe
    df1_matched = pd.concat(df1_matched) 
    # df1_unmatched = pd.concat(df1_unmatched) 

    return df1_matched, df1_unmatched

def intersectioned_grains(match_grainsID_L2,match_grainsID_L2_n,match_grainsID_L3):
    #%% Finding out common/intersection of grains bw load L12 and L23
    # match_grainsID_L123 = list(set(match_grainsID_L2_n).intersection(match_grainsID_L2))
    match_grainsID_L123 = set_intersection(match_grainsID_L2_n,match_grainsID_L2)
    print(f"Final mathced grains IDs-Load123: {len(match_grainsID_L123)}")

    #%% Seelcting out corresponding grains IDs from "match_grainsID_L3" based on "Grains ID index of L2"
    common_grains_L3 = common_matched_grains(match_grainsID_L123,match_grainsID_L2_n,match_grainsID_L3)
    return common_grains_L3

