'''
Created: Friday, July 28 2023
Author: Amit 

#%%%%% Readme #%%%%%
This script has three section: 
1st section: genrate new grains IDs and then creat a map (dictionary) of the old and new grains IDs, and saved as parquet files
2nd section: Select out grains between the layers: trimmed and dropped grains by using given "z_offset"
3rd section: To merge three layer data set together to make eack load step as data set as one set for better comparison of data
             This 3rd section is incomplete, to to this merger new script "grains_bw_layers.py" is written. 

Note#1: As per the current code status, "section-1" code is only required which does add "new_grainIDs". Also, "Section-2" of the code does not require since 
that just selecting of the trimmed grains. But if require we can do run that too, but modification is required. 
Note#2: Caution: we need to check for "Z" values i.e "Z_offset", since each layer data-set from HEDM come with same domain of X-Y-Z, to make each layer
        distingushable, we need to add/subtarct "offset values" based on beam-thickness. So, this has to be checked and if required, do correct the "Z_offset"

'''
# Code for Grain_indexing #
# Importing Libraries
import os
import glob
from turtle import shape
import pandas as pd
import time
import numpy as np
import json
import matplotlib.pyplot as plt

#%%%%%%%%%%%% time tic    %%%%%%%%%%%%%%%
tic = time.time()

#%%%%%%%%%%%% Functions  %%%%%%%%%%%%%%%
def grainIDs_map(newID,oldID):
    grain_map_dict = {}
    grain_map_dict['newID'] = newID
    grain_map_dict['oldID'] = oldID
    return grain_map_dict

def write_json(dict_data, json_file, indent=4):
    with open(json_file, 'w') as fp:
        json.dump(dict_data, fp, indent=indent)
    return None

def trim_drop_IDs_map(trimmedID,droppedID):
    trimmed_grainIDs_dic = {}
    trimmed_grainIDs_dic['Trimmed_grainsID'] = trimmedID
    trimmed_grainIDs_dic['Dropped_grainsID'] = droppedID
    return trimmed_grainIDs_dic

#%%%%%%%%%%%% Functions END %%%%%%%%%%%%%%%

#0. Defining folder path and listing all the files
#pth = os.getcwd()
pth = '/home/asingh38/projects/nsf_twinning_2022'
sam_n = eval(input("Enter the sample-number:"))

# ''' 
#%%%%%%%%%%%% Section 1st %%%%%%%%%%%%%%%
fles  = glob.glob(f'{pth}/raw_data/sample_{sam_n}/*.csv')
# fles  = glob.glob(f'{pth}/raw_data/sample_{sam_n}/sam3_ff_290N*.csv')# for just one load-step
fles.sort()
#1.2. Importing files
new_grainsID = []
old_grainsID = []

for fle in fles:
    fls_name = os.path.basename(fle) # To drop the path/directory name, only file name will give
    print(f"file name:",fls_name)
    splt_fls = fls_name.split('_')
    sample = splt_fls[0][-1]
    load = splt_fls[2][:-1]
    layer = splt_fls[3][-1]
    
    #1.1 Reading files
    data = pd.read_csv(fle)
    headers = data.head(7)

    headers.to_csv(f'{pth}/raw_data/sample_{sam_n}/txt_files/sam{sample}_{int(load):03d}N_Layer{layer}_headers.txt', sep ='\t', index = False)        
    
    #1.2 reading data
    data = pd.read_csv(fle,skiprows=(range(8)),sep ='\t', index_col = False)
    data = data.rename(columns={'%GrainID':'GrainID'})
    old_ids = data['GrainID'].tolist()
    old_grainsID.append(old_ids)

    #1.3 writing a grainID name
    new_ids = []
    for index, row in data.iterrows():
        # new_id = int(f'{sample}{int(load):03d}{layer}0{index + 1:05d}')
        new_id = int(f'{sample}{layer}{int(load):03d}{index + 1:05d}') # for sample3
        new_ids.append(new_id)
    new_grainsID.append(new_ids)

    #1.4 Saving files: new 'Data'
    # data.insert(loc = 0, column = 'new_GrainID', value = new_ids)
    data.index = new_ids
    data.index.name = 'new_GrainID' # This is just to make 'new_GrainID' as a indexer, instead of default one
    data.to_parquet(f'{pth}/raw_data/sample_{sam_n}/parquet_files/sam{sample}_{int(load)}N_Layer{layer}_grains.parquet', index = True)
    # data.to_csv(f'{pth}/raw_data/sample_{sam_n}/csv_files/sam{sample}_{int(load):03d}N_Layer{layer}_grains_new.csv', index = True)

# #1.5 Constructing 'grainID_map'
# grains_IDs_map = grainIDs_map(new_grainsID,old_grainsID)
# save_map = f'{pth}/raw_data/sample_{sam_n}/txt_files/sam{sample}_all_grains_IDs_new.json'
# write_json(grains_IDs_map,save_map)
# ## To save as "dictionary" format in txt file
# grains_IDs_map = pd.DataFrame(grains_IDs_map)
# grains_IDs_map.to_csv(f'{pth}/raw_data/sample_{sam_n}/txt_files/sam{sample}_all_grains_IDs.txt',sep ='\t', index = False)
### To save as normal format in txt file
# with open(f'{pth}/raw_data/sample_{sam_n}/txt_files/sam{sample}_all_grains_IDs.txt', 'w') as f:
#     for key, value in grains_IDs_map.items():
#         f.write(f"{key}\t{value}\n")


# '''

'''
#%%%%%%%%%%%% Section 2nd %%%%%%%%%%%%%%%
##### Selecting out grains between the layers: trimmed and dropped grains #### 
z_offset = eval(input(f"Enter the Z-offset(thickness of layer):")) # 200 for sample-1 and sample-3, and 400 for sample-2
#2.1 IMPORTING DATA FROM PARQUET FILE
fls_p  = glob.glob(f'{pth}/raw_data/sample_{sam_n}/parquet_files/*.parquet')
fls_p.sort()

trim_grainsID = []
drop_grainsID = []
#2.2 Reading file
for fle_p in fls_p:
    #2.3 selecting out sample details:
    fn = os.path.basename(fle_p) # To drop the path/directory name, only file name will give
    splt_fls = fn.split('_')
    sample = splt_fls[0][-1]
    load = splt_fls[1][:-1]
    layer = splt_fls[2][-1]
    #2.4 reading data
    df = pd.read_parquet(fle_p)
    df = df.reset_index() # This was done to remove the 'new_GrainID' as a indexer
    #2.5 trimming the grains out side the bound
    df_trim = df[df['Z'].between(-z_offset/2, z_offset/2, inclusive='both')]
    trim_grain = df_trim['new_GrainID'].tolist()
    trim_grainsID.append(trim_grain)
    # df_trim.to_parquet(f'{pth}/raw_data/sample_{sam_n}/parquet_files/trimmed_grains/sam{sample}_{int(load):03d}N_Layer{layer}_grains_trimmed.parquet', index = True)
    # df_trim.to_csv(f'{pth}/raw_data/sample_{sam_n}/csv_files/trimmed_grains/sam{sample}_{int(load):03d}N_Layer{layer}_grains_trimmed.csv', index = True)
    # x = df_trim ['X'].values
    # y = df_trim ['Y'].values
    # z = df_trim ['Z'].values
    # plt.scatter(x,z, marker='H', edgecolor='none', s=20, alpha=0.5) # for 2D plot
    # plt.show()
    #2.6 the grains inside the offset
    df_drop = df[~df['Z'].between(-z_offset/2, z_offset/2, inclusive='both')] # here '~' is used just to "oppsite" of above
    drop_grain = df_drop['new_GrainID'].tolist()
    drop_grainsID.append(drop_grain)
    # x_out = df_drop ['X'].values
    # y_out = df_drop ['Y'].values
    # z_out = df_drop ['Z'].values
    # plt.scatter(x_out,z_out, marker='D', edgecolor='none', s=20, alpha=0.5) # for 2D plot
    # plt.show()
    # df_drop.to_parquet(f'{pth}/raw_data/sample_{sam_n}/parquet_files/dropped_grains/sam{sample}_{int(load):03d}N_Layer{layer}_grains_dropped.parquet', index = True)
#2.7 condtructing map for trimmed_dropped grains IDs
trim_drop_IDs = trim_drop_IDs_map(trim_grainsID,drop_grainsID)
save_map = f'{pth}/raw_data/sample_{sam_n}/txt_files/sam{sample}_trim_drop_grains_IDs.json'
write_json(trim_drop_IDs,save_map)


   # z_coo = data_p['Z'].values # values is used just to have the values
    # print(len(z_coo))
    # new_z_coo = []
    # for z in z_coo:
    #     if -1*z_offset/2 <= z <= z_offset/2:
    #         new_z_coo.append(z)
    # print(len(new_z_coo))
'''

