''' 
Date: 24th October 2024
Author: Amit, ACDM Lab
Description: This is the library for all the function related to 
             strain energy calculation

'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-0: Loading required python bulild-in libraries and functions 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np 
from lib_functions_analysis import *
from lib_mathematics import list_to_matrix_nxn
from lib_crystal_mechanics import symmetric_matrix_to_voigt_array
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-1: Basic general function related to "strain-ananlysis" i.e. 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% 1.1
def normalized_strainenergy(U):
    U_norm = []
    for u in U:
        U_norm.append(u/U[0])
    return U_norm

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-2: Functions related to strain-energy (elastic) calculation, based on synchrotron data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% 2.0 
def strain_energy_density_estimation(pth, sam_n, ids, load_c, load_f, icrmt, rad_factor, C_matrix):
    counter = 0
    strain_energy_cf =[]
    while load_c<=load_f:
        # Reading the modified data for each load step
        df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N.csv",skiprows=1, sep = "\t")
        df_modif = df_modif.set_index(["%new_GrainID"])
        #1 -------------------------------------------------------------------
        #%% Orientation matrix
        rot = df_modif.loc[ids[counter]]["O11":"O33"]
        rot = list_to_matrix_nxn(rot)
        #2 -------------------------------------------------------------------
        #%% Strain-matrix is in sample frame as per received data
        epsln_sf = df_modif.loc[ids[counter]]["eKen11":"eKen33"]/1000 # 1000 is devided because strain data is in micron
        epsln_sf = list_to_matrix_nxn(epsln_sf)
        # epsln_sf_voigt = symmetric_matrix_to_voigt_array(epsln_sf, "strain") # in voigt notation
        #3 -------------------------------------------------------------------
        #%% Converting Strain to Crystal Frame from Sample frame
        epsln_cf = np.matmul(rot.T, np.matmul(epsln_sf,rot)) ## cf = crsytal frame
        # Strain in Voigt Notation
        epsln_cf_voigt = symmetric_matrix_to_voigt_array(epsln_cf, "strain") # in cf =crystal frame
        #4 -------------------------------------------------------------------
        #%% Stress in Voigt notation
        sigma_cf_voigt = np.matmul(C_matrix, epsln_cf_voigt) # in cf =crystal frame
        # Stress in matrix form in crytal form
        # sigma_cf = voigt_array_to_symmetric_matrix(sigma_cf_voigt, "stress")

        ## Converting stress from "CF" to sample-frame
        # sigma_sf = np.matmul(rot, np.matmul(sigma_cf, rot.T))
        # sigma_sf_voigt = symmetric_matrix_to_voigt_array(sigma_sf, "stress"), # in voigt notation
        #5 ------------------------------------------------------------------
        ## Grain-radius
        # rad = df_modif.loc[ids[counter]]["GrainRadius"]*rad_factor
        # vol = (4/3)*np.pi*rad**3
        #6 -------------------------------------------------------------------
        ## Strain energy in CF
        strain_density = (0.5*np.dot(sigma_cf_voigt.T, epsln_cf_voigt))
        strain_energy_cf.append(strain_density)
        ## Strain energy in SF
        # strain_energy_sf.append(0.5*np.dot(sigma_sf_voigt.T, epsln_sf_voigt))

        #7 ------------------------------------------------------------------
        load_c += icrmt
        counter += 1
    return strain_energy_cf

#%% 2.1-
def strain_energy_for_load_step(pth, sam_n, list_of_pg_Ids, compare, load_f, icrmt, C_matrix):
    '''This function will give the strain energy
       Input: compare: wheather comapring based on load_step from initial or not
              C_matrix = Siffness matrix
       Output: Strain_energy
    '''
    C_Ti = C_matrix
    same_pg = list_of_pg_Ids
    #-----------------------------------------
    strain_energy_pg = []
    counter = 0
    if compare == 'step_0':
        load_c = 0
        while load_c <=load_f:
            # Reading the modified data for each load step
            df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N.csv",skiprows=1, sep = "\t")
            df_modif = df_modif.set_index(["%new_GrainID"])
            ## For parent grain
            #%% Orientation matrix
            rot_pg = df_modif.loc[same_pg[counter]]["O11":"O33"]
            rot_pg = list_to_matrix_nxn(rot_pg)

            #%% Strain-matrix
            epsln_sf_pg = df_modif.loc[same_pg[counter]]["eKen11":"eKen33"]/1000
            epsln_sf_pg = list_to_matrix_nxn(epsln_sf_pg)
            epsln_cf_voigt_pg = symmetric_matrix_to_voigt_array(epsln_cf_pg, "strain") # in cf =crystal frame
            epsln_cf_pg = np.matmul(rot_pg.T, np.matmul(epsln_sf_pg,rot_pg)) ## cf = crsytal frame

            ## Stress calculation
            sigma_cf_voigt_pg = np.matmul(C_Ti, epsln_cf_voigt_pg)
            ## Strain energy
            u_cf_pg = 0.5*np.dot(sigma_cf_voigt_pg.T, epsln_cf_voigt_pg)
            strain_energy_pg.append(u_cf_pg)
            #__________________________________________________________
            load_c += icrmt
            counter += 1
    return strain_energy_pg

#%% 2.2-
def relative_strain_energy_change(load_s, twin_n, U):
    '''
    Inputs: twin_n = List of twin_numbers of a particular load-step in twin, e.g load_25 has 8 twins
            U = Dictionary of the energies of the parent-grain based on that particular load-step and number of twin
            load_s = current load_step
            load_f = Final load_step
    Outputs: dU = Dictonary of the "Change in the energy" for that particular load_step as per the number of twins
    '''
    UE_pg = U
    dUE_pg = {} #Dictionaries of the change in the strain-energies
    for ii, twn in enumerate(twin_n, start =1):
        u_pg = UE_pg[f"LS_{load_s}_PG_{ii}"]
        du = []
        for i in range(len(u_pg)-1):
            du.append((u_pg[i+1]-u_pg[i])*100/u_pg[i]) # dU = U_(i+1)-U_(i)/U_(i)
        dUE_pg[f"LS_{load_s}_PG_{ii}"] = du
    return dUE_pg

#%% 2.3-
def average_relative_strain_energy_change(load_s, load_f, twin_n, dU):
    '''
    Inputs: twin_n = List of twin_numbers of a particular load-step in twin, e.g load_25 has 8 twins
            dU = Dictionary of the energies of the parent-grain based on that particular load-step and number of twin
            load_s = current load_step
            load_f = Final load_step
    Outputs: avg_dU, std_dU = list of average and standard deviation of strain energy change
    '''
    dUE_pg  = dU
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    load_step_1 = 25 # very 1st loading step, with respect to this everything is calculated
    n_load_steps = int((load_f-load_s)/load_step_1) + 1
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    avg_dU_ls,std_dU_ls = {},{}
    for i in range(n_load_steps):
        dU_ls = []
        for ii, twn in enumerate(twin_n, start =1):
            dU_ls.append(dUE_pg[f"LS_{load_s}_PG_{ii}"][i])
        avg_dU_ls[f"LS_{i+1}"] = np.average(dU_ls)
        std_dU_ls[f"LS_{i+1}"] = np.std(dU_ls)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    avg_dU, std_dU = [], []
    for i in range(n_load_steps):
        avg_dU.append(avg_dU_ls[f"LS_{i+1}"])
        std_dU.append(std_dU_ls[f"LS_{i+1}"])
    return avg_dU, std_dU 

#%% 2.4-
def normalized_energy(load_s, twin_n, U, method):
    '''
    Method:: type "parentgrain" for parent-grain
              or "twin" for twin
    '''
    UE = U
    dUE = {} #Dictionaries of the change in the strain-energies
    if method == "parentgrain":
        for ii, twn in enumerate(twin_n, start =1):
            u = UE[f"LS_{load_s}_PG_{ii}"]
            du = []
            for i in range(len(u)-1):
                du.append((u[i])/u[0]) # dU = U_(i+1)-U_(i)/U_(i)      
        dUE[f"LS_{load_s}_PG_{ii}"] = du
    elif method == "twin":
        print(len(twin_n))
        for jj, twn in enumerate(twin_n, start =1):
            u = UE[f"LS_{load_s}_Twin_{jj}"]
            du = []
            for i in range(len(u)-1):
                du.append((u[i])/u[1]) # dU = U_(i+1)-U_(i)/U_(i)
        dUE[f"LS_{load_s}_Twin_{ii}"] = du
    return dUE

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-3: Functions related to "stress_eq, strain_eq, and enegy_density" for all "mathced grains as "parent-grains" across laod-steps
# and `matched-grains' from unmatched grains as "potential-twins" calculation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def stress_strain_energy_density_for_all_matched_unmatched_grains(pth, sam_n, Id_list, compare, load_c, rad_factor, C_matrix):
    '''
    Description: This function does the stress, strain equivalent calculation either in sample/crystal frame
                 In addition, it will also calculate the "energy-density" but in "Crystal-frame"
    Input: Id_list: list of the grains Ids, either for final-matched or "matched-grains from unmatched grains
            load_c: current load-steps i.e. from which one want to etsimate this
            C_matrix: The stiffness matrix of 6x6 dimensions
    Output: Stress_eq, Strain_eq either in sample/crystal frame and U in only crystal frame
    '''
    C_Ti = C_matrix
    # strain_eq_pg, stress_eq_pg = [], []
    # sigma_33, epsln_33 = [], []
    # sigma_22, epsln_22 = [], []
    # sigma_11, epsln_11 = [], []
    U_pg = []
    vol_pg = []
    #%%------------------------------------------------------------------------------
    if compare == 'step_0':
        for id in Id_list:
            # Reading the modified data for each load step
            df_modif = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N.csv",skiprows=1, sep = "\t")
            df_modif = df_modif.set_index(["%new_GrainID"])
            #-------------------------------------------------------------------
            #%% Orientation matrix
            rot_pg = df_modif.loc[id]["O11":"O33"]
            rot_pg = list_to_matrix_nxn(rot_pg)
            #-------------------------------------------------------------------
            #%% Strain-matrix in Sample-frame as per as received data
            epsln_sf_pg = df_modif.loc[id]["eKen11":"eKen33"]
            epsln_sf_pg = list_to_matrix_nxn(epsln_sf_pg)
            #-------------------------------------------------------------------
            #%% Converting Strain to Crystal Frame then in Voigt notation form
            epsln_cf_pg = np.matmul(rot_pg.T, np.matmul(epsln_sf_pg,rot_pg)) ## cf =crsytal frame
            #&& Stress components
            # epsln_33.append(epsln_cf_pg[2,2])
            # epsln_22.append(epsln_cf_pg[1,1])
            # epsln_11.append(epsln_cf_pg[0,0])
            #%% Equivalent strain calculation
            # strain_eq_pg.append(np.sqrt((2/3)*doubledot(epsln_sf_pg,epsln_sf_pg))) # This is in sample frame

            # strain_eq_pg.append(np.sqrt((2/3)*doubledot(epsln_cf_pg,epsln_cf_pg))) # This is in Crystal-frame  

            # Strain in Voigt Notation form using "CF"
            epsln_cf_voigt_pg = symmetric_matrix_to_voigt_array(epsln_cf_pg, "strain")# when epsln is not in crystal frame
            #------------------------------------------------------------------------------------------
            #%% Stress in crystal-frame using Voigt notation
            sigma_cf_voigt_pg = np.matmul(C_Ti, epsln_cf_voigt_pg) ## This is an array of shape (6,1) i.e 6-row and 1-column
            # # sigma_22.append(sigma_cf_voigt_pg[1]) # Sigma-22
            # #% Converting stress (which is in CF) as equivalent to matrix form
            # sigma_voigt_shear_double = np.array([sigma_cf_voigt_pg[0], sigma_cf_voigt_pg[1], sigma_cf_voigt_pg[2], 
            #                                     sigma_cf_voigt_pg[3]*(2**0.5), sigma_cf_voigt_pg[4]*(2**0.5), sigma_cf_voigt_pg[5]*(2**0.5)])
            # #% Calculating stress_eq in CF
            # sigma_eq_pg.append(np.sqrt((3/2)*doubledot(sigma_voigt_shear_double,sigma_voigt_shear_double)))
            ## Converting stress from Voigt to matrix form,note voigt notation is in "CF" so converted stress-matrix will be CF
            # sigma_cf_pg = voigt_array_to_symmetric_matrix(sigma_cf_voigt_pg, "stress")
            #-------------------------------------------------------------------------------------------
            #%% Converting stress from "CF" to sample-frame i.e. "SF"
            # sigma_sf_pg = np.matmul(rot_pg, np.matmul(sigma_cf_pg, rot_pg.T))
            
            # sigma_33.append(sigma_cf_pg[2,2])
            # sigma_22.append(sigma_cf_pg[1,1])
            # sigma_11.append(sigma_cf_pg[0,0])
            # ##% Calculating stress_eq in SF
            # # stress_eq_pg.append(np.sqrt((3/2)*doubledot(sigma_sf_pg,sigma_sf_pg)))
            # ##% Calculating stress_eq in CF
            # stress_eq_pg.append(np.sqrt((3/2)*doubledot(sigma_cf_pg,sigma_cf_pg)))

            rad = df_modif.loc[id]["GrainRadius"]*rad_factor
            vol_pg.append(((4/3)*np.pi*rad**3))

            ## Strain energy in CF
            U_pg.append((0.5*np.dot(sigma_cf_voigt_pg.T, epsln_cf_voigt_pg)))
    else:
        print(f" Opps, this is for step_0 only!!")

    return  U_pg, vol_pg #stress_eq_pg, strain_eq_pg, sigma_33, epsln_33, sigma_22, epsln_22, sigma_11, epsln_11