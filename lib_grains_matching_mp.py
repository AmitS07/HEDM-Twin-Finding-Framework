'''
Date: Thursday Sept. 25 2025
Author: Amit(amitdravid@outlook.com), ACDM Lab, Tuscaloosa
Description: This library contaion function for grains matching,across load wise. 
             This does analysis in parallel i.e. multiprocessing function of Python

'''
#%% IMPORTING LIBRAIES
import pandas as pd

# importing manually defined functions 
from lib_orientations_symmetries import *
from lib_functions_analysis import *
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%5.2.2 Misorienation and Euclidean dist calculation between two-consecutive load-steps using multiprocessing
def compute_pair(args):
    l1_idx, l1_vals, df2 = args
    rot1 = rot2mat(l1_vals["O11":"O33"].to_list())
    q1 = rot2quat_scipy(rot1)
    cord1 = l1_vals[['X', 'Y', 'Z']]

    results = []
    for j, (l2_idx, l2_vals) in enumerate(df2.iterrows()):
        rot2 = rot2mat(l2_vals["O11":"O33"].to_list())
        q2 = rot2quat_scipy(rot2)
        cord2 = l2_vals[['X', 'Y', 'Z']]
        eucid_dist = euclidean_dist(cord2, cord1)
        misori_l12 = mis_ori(q1, q2)
        results.append((l2_idx, [round(misori_l12, 5), round(eucid_dist, 5)]))
    return (l1_idx, results)

def calc_misori_euclideanDist_bw_twoLoad_mp(df1, df2, n_processor):
    import multiprocessing as mp
    grainID1 = [str(idx) for idx in df1.index]
    grainID2 = [str(idx) for idx in df2.index]
    with mp.Pool(processes = n_processor) as pool:
        results = pool.map(compute_pair,[(l1_idx, l1_vals, df2) for l1_idx, l1_vals in df1.iterrows()])

    # Reconstruct matrix
    grains_matrix = pd.DataFrame(index=grainID1, columns=grainID2, dtype=object)
    for l1_idx, row_results in results:
        for l2_idx, values in row_results:
            grains_matrix.loc[str(l1_idx), str(l2_idx)] = values

    return grains_matrix