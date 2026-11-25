''' 
Date: 24th March 2025
Author: Amit, ACDM Lab
Descripsion: This set of code-lines/functions to find out the -
            "Neighbouring" grains to the parent-grains which "twinned"
'''

#%%----------------- Importing inbuild-python Libraries ----------------#%%
import pandas as pd
from lib_functions_analysis import euclidean_dist

#%%------------------ Defining Functions -----------------#%%
def neighbour_grains(pth, sam_n, load_c, twin_pgs, remain_pgs, rad_factor, tol_rad, neighbr_grains_IDs, method):
    '''
    Objective: This is to find out the neighboring-grains to the parent-grains which got twinned,
                based on given threshold tolerance and criterion
    Input: The major inputs are-
            all_pgs =  lists of all the 'final matched-grains'
            twin_pgs = list of all the 'Parent-grains which are twin'
            tol_rad = Radius tolerance, due distortion of the shape
            rad_factor = Factor of radius, calculated based on volume occupied 
                        by all the garins and measured volume
            neighbr_grains_IDs = empty dictionary need to define, before using this funtion
            method = Whether we want to seach for just 6-neighbors or all neighrs based on criteria 
                        if just 6-neighbors then, method = "6-neighbors", else put method = "all"
    ORDER of INPUT:: pth, sam_n, load_c, twin_pgs, remain_pgs, rad_factor, tol_rad, neighbr_grains_IDs
    #---------------------------------------------------------------------------------------------------
    Output: list of the neighbor-grains to "a" parent-grain which got twin, 
    ORDER of OUTPUT: rad_pgtw, xyz_pgtw, rad_pg, xyz_pg, neighbr_grains_IDs
    '''
    xyz_pgtw, xyz_pg = [],[]
    rad_pgtw, rad_pg =[], []
    if method == "6-neighbors":
        for tpg in twin_pgs:
            neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"] = [] # To make sure, Dictionay can take data in list form
            df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N.csv",skiprows=1, sep = "\t")
            df_modif = df_modif.set_index(["%new_GrainID"])
            # Selecting Parent-Grains which got twinned, based on Twin-IDs
            cord_pgtw = df_modif.loc[tpg]["X":"Z"]
            r_pgtw = df_modif.loc[tpg]["GrainRadius"]*rad_factor
            xyz_pgtw.append(cord_pgtw)
            rad_pgtw.append(r_pgtw)
            # Selecting Parent-Grains which did not gettwinned, hence could be Neighboring-grains
            for pg in remain_pgs:
                cord_pg = df_modif.loc[pg]["X":"Z"]
                r_pg = df_modif.loc[pg]["GrainRadius"]*rad_factor
            # Doing Neighboring Grains Analysis, Criteria
                eucid_dist = euclidean_dist(cord_pgtw, cord_pg)
                if r_pg<eucid_dist<=((r_pgtw+r_pg)+tol_rad):
                    neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"].append((pg, eucid_dist))
            #%% Checking for minimum-6 neighboring_grains
            if len(neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"])>=6: #% Check, whether number of neighbor grains are
                neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"].sort(key=lambda x: x[1]) # Sorting in acending order based on 2nd value i.e. euclidean dist
                neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"] = neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"][:6] # just to select closest neighbor 6-grains
                print(f"Closest 6-neihbour grains for a twin-PG at load-{load_c}:",{len(neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"])})
                for pg, dist in neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"]:
                    cord_pg = df_modif.loc[pg][["X", "Y", "Z"]].values
                    r_pg = df_modif.loc[pg]["GrainRadius"]*rad_factor
                    xyz_pg.append(cord_pg)
                    rad_pg.append(r_pg)
            else:
                print(f"Opps!Neighbour grains are not more than 6, these are just {len(neighbr_grains_IDs[f'Load_{load_c}_pgtw_{tpg}'])}")
    else:
         for tpg in twin_pgs:
            neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"] = [] # To make sure, Dictionay can take data in list form
            df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N.csv",skiprows=1, sep = "\t")
            df_modif = df_modif.set_index(["%new_GrainID"])
            # Selecting Parent-Grains which got twinned, based on Twin-IDs
            cord_pgtw = df_modif.loc[tpg]["X":"Z"]
            r_pgtw = df_modif.loc[tpg]["GrainRadius"]*rad_factor
            xyz_pgtw.append(cord_pgtw)
            rad_pgtw.append(r_pgtw)
            # Selecting Parent-Grains which did not gettwinned, hence could be Neighboring-grains
            for pg in remain_pgs:
                cord_pg = df_modif.loc[pg]["X":"Z"]
                r_pg = df_modif.loc[pg]["GrainRadius"]*rad_factor
            # Doing Neighboring Grains Analysis, Criteria
                eucid_dist = euclidean_dist(cord_pgtw, cord_pg)
                if r_pg<eucid_dist<=((r_pgtw+r_pg)+tol_rad):
                    neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"].append((pg, eucid_dist))
                    xyz_pg.append(cord_pg)
                    rad_pg.append(r_pg)
            print(f"#Neihbour grains for twins at load-{load_c}:",{len(neighbr_grains_IDs[f"Load_{load_c}_pgtw_{tpg}"])})
    return rad_pgtw, xyz_pgtw, rad_pg, xyz_pg, neighbr_grains_IDs #r_pgtw, cord_pgtw, r_pg, cord_pg





        





