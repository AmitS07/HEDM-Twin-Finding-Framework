''' 
Date: Thursday, 28th August 2025
Author: Amit, ACDM Lab
Descripsion: This library contains the functions that are related to crystal-plasticity mechanics
'''
#%% Importing python libraries 
import numpy as np

from lib_mathematics import *
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Part#1: Functions to calculate Schimd-tensors for HCP slip and twins systems
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hcp_slip_schmid_tensor(ctype, covera = 1.633):
    import sys
    '''
    input:: ctye- crystal-type, "hcp"
            covera = ratio of c/a, default is 1.633, 
    Output:: pij = Schimd-factor
    '''
    # Define n and d arrays
    if (ctype == 'hcp'):
        n = np.double(np.array([ \
            # Basal slip planes
            [ 0,  0,  0,  1], \
            [ 0,  0,  0,  1], \
            [ 0,  0,  0,  1], \
            # Prismatic slip planes
            [ 1, -1,  0,  0], \
            [ 1,  0, -1,  0], \
            [ 0,  1, -1,  0], \
            # Pyramidal-I <a> slip planes
            [ 1, -1,  0,  1], \
            [-1,  1,  0,  1], \
            [ 1,  0, -1,  1], \
            [-1,  0,  1,  1], \
            [ 0, -1,  1,  1], \
            [ 0,  1, -1,  1], \
            # Pyramidal-I c+a slip planes
            [ 1, -1,  0,  1], \
            [ 1, -1,  0,  1], \
            [-1,  1,  0,  1], \
            [-1,  1,  0,  1], \
            [ 1,  0, -1,  1], \
            [ 1,  0, -1,  1], \
            [-1,  0,  1,  1], \
            [-1,  0,  1,  1], \
            [ 0,  1, -1,  1], \
            [ 0,  1, -1,  1], \
            [ 0, -1,  1,  1], \
            [ 0, -1,  1,  1] \
        ]))
        d = np.double(np.array([ \
            # Basal slip directions
            [ 1,  1, -2,  0], \
            [-1,  2, -1,  0], \
            [-2,  1,  1,  0], \
            # Prismatic slip directions
            [ 1,  1, -2,  0], \
            [-1,  2, -1,  0], \
            [-2,  1,  1,  0], \
            # Pyramidal-I <a> slip directions
            [ 1,  1, -2,  0], \
            [-1, -1, -2,  0], \
            [-1,  2, -1,  0], \
            [ 1, -2,  1,  0], \
            [-2,  1,  1,  0], \
            [ 2, -1, -1,  0], \
            # Pyramidal-I <c+a> slip directions
            [-1,  2, -1,  3], \
            [-2,  1,  1,  3], \
            [ 1, -2,  1,  3], \
            [ 2, -1, -1,  3], \
            [-1, -1,  2,  3], \
            [-2,  1,  1,  3], \
            [ 1,  1, -2,  3], \
            [ 2, -1, -1,  3], \
            [-1, -1,  2,  3], \
            [ 1, -2,  1,  3], \
            [-1,  2, -1,  3], \
            [ 1,  1, -2,  3] \
            ]))
    else:
        sys.exit('Error: Crystal type not supported.')
    # Find number of slip systems
    N = np.shape(n)[0]
    # If HCP, first convert to Cartesian
    if (ctype == 'hcp'):
        ntrans = np.zeros([4, 3])
        ntrans[0, 0] = 1
        ntrans[0, 1] = 1 / np.sqrt(3)
        ntrans[1, 1] = 2 / np.sqrt(3) 
        ntrans[3, 2] = 1 / covera
        ntmp = np.matmul(n, ntrans)
        n = np.copy(ntmp)
        dtrans = np.zeros([4, 3])
        dtrans[0, 0] = 1.5
        dtrans[0, 1] = np.sqrt(3) / 2
        dtrans[1, 1] = np.sqrt(3)
        dtrans[3, 2] = covera
        dtmp = np.matmul(d, dtrans)
        d = np.copy(dtmp)
    # Normalize the slip normals and slip directions
    for i in range(0, N):
        n[i, :] = n[i, :] / np.linalg.norm(n[i, :])
        d[i, :] = d[i, :] / np.linalg.norm(d[i, :])    
    # # Define symmetric portions of the Schmid tensors
    pij = []
    for ii,jj in zip(n,d):
        pij.append(np.outer(ii,jj))
    return pij

def hcp_twin_schmid_tensor(ctype, covera = 1.633):
    import sys
    '''
    input:: ctye- crystal-type, "hcp"
            covera = ratio of c/a, default is 1.633, 
    Output:: pij = Schimd-factor
    '''
    # Define n and d arrays
    if (ctype == 'hcp'):
        n = np.double(np.array([ \
            # Tensile twin (T1) planes
            [ 1,  0, -1,  2], \
            [ 0,  1, -1,  2], \
            [-1,  1,  0,  2], \
            [-1,  0,  1,  2], \
            [ 0, -1,  1,  2], \
            [ 1, -1,  0,  2], \
             # Tensile twin (T2) planes
            [ 1,  1, -2,  1], \
            [ 1, -2,  1,  1], \
            [-2,  1,  1,  1], \
            [-1, -1,  2,  1], \
            [ 2,  1, -1,  1], \
            [-1,  2, -1,  1], \
            ]))
        d = np.double(np.array([ \
            # Tensile twin (T1) direction 
            [-1,  0,  1,  1], \
            [ 0, -1,  1,  1], \
            [ 1, -1,  0,  1], \
            [ 1,  0, -1,  1], \
            [ 0,  1, -1,  1], \
            [-1,  1,  0,  1], \
             # Tensile twin (T2) direction
            [-1, -1,  2,  6], \
            [-1,  2, -1,  6], \
            [ 2, -1, -1,  6], \
            [ 1,  1, -2,  6], \
            [-2, -1,  1,  6], \
            [ 1, -2,  1,  6], \
            ]))
    else:
        sys.exit('Error: Crystal type not supported.')
    # Find number of slip systems
    N = np.shape(n)[0]
    # If HCP, first convert to Cartesian
    if (ctype == 'hcp'):
        ntrans = np.zeros([4, 3])
        ntrans[0, 0] = 1
        ntrans[0, 1] = 1 / np.sqrt(3)
        ntrans[1, 1] = 2 / np.sqrt(3) 
        ntrans[3, 2] = 1 / covera
        ntmp = np.matmul(n, ntrans)
        n = np.copy(ntmp)
        dtrans = np.zeros([4, 3])
        dtrans[0, 0] = 1.5
        dtrans[0, 1] = np.sqrt(3) / 2
        dtrans[1, 1] = np.sqrt(3)
        dtrans[3, 2] = covera
        dtmp = np.matmul(d, dtrans)
        d = np.copy(dtmp)
    # Normalize the slip normals and slip directions
    for i in range(0, N):
        n[i, :] = n[i, :] / np.linalg.norm(n[i, :])
        d[i, :] = d[i, :] / np.linalg.norm(d[i, :])    
    # # Define symmetric portions of the Schmid tensors
    pij = []
    for ii,jj in zip(n,d):
        pij.append(np.outer(ii,jj))
    return pij
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Part-2: Functions related resolved shear stress calculation (RSS)
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##2.1 for slip systems
def rss_hcp_all_slip_a_grain(schmid_tensor, sigma_cf_gr):
    ''' 
    Input: Schimd Tensor (P = n*d), and 
            Stress-tensor, both are 3*3 matrix of the grain
    Output: RSS = sigma_tensor*Schimd_tensor
    This function does RSS calculation of total 24 active slip-systems (Basal, Prismatic, Pyramidal) for HCP crystals 
    The Schimd tensor is a product of slip planes and directions as P = n*d
    '''
    Pij_scys = schmid_tensor
    ## Estimating Tau-RSS for all 24 possible slip-system i.e Basal(3), Psimatic(3), PyramidalI_a(6) and PyramidalI_ca(12)
    rss_sl_basal,rss_sl_prism,rss_sl_pyramI_a, rss_sl_pyramI_ca = [], [], [], []
    #%% RSS calculations
    rss_sl_basal.append(abs(np.array([doubledot(sigma_cf_gr,jj) for jj in Pij_scys[0:3]])))
    rss_sl_prism.append(abs(np.array([doubledot(sigma_cf_gr,jj) for jj in Pij_scys[3:6]])))
    rss_sl_pyramI_a.append(abs(np.array([doubledot(sigma_cf_gr,jj) for jj in Pij_scys[6:12]])))
    rss_sl_pyramI_ca.append(abs(np.array([doubledot(sigma_cf_gr,jj) for jj in Pij_scys[12:24]])))
    return rss_sl_basal,rss_sl_prism,rss_sl_pyramI_a, rss_sl_pyramI_ca

##2.2 for twin systems
def rss_hcp_all_tensiletwin_a_grain(schmid_tensor, sigma_cf_gr):
    ''' 
    Input: Schimd tensor (P = n*d), and Stress-tensor, bith are 3*3 matrix
    Output: RSS = sigma_tensor*Schimd_tensor
    This function does RSS calculation of total 12 active twin-systems (Tensile-T1 twins, Tensile-T2 twins) for HCP crystals 
    The Schimd tensor is a product of slip planes and directions as P = n*d
    '''
    Pij_hcp_twin = schmid_tensor
    ## Estimating Tau-RSS for all 6 possible twin-system i.e T1(3), T2(3) 
    rss_tw_t1,rss_tw_t2 = [], []
    #%% RSS calculations
    rss_tw_t1.append(abs(np.array([doubledot(sigma_cf_gr,jj) for jj in Pij_hcp_twin[0:6]])))
    rss_tw_t2.append(abs(np.array([doubledot(sigma_cf_gr,jj) for jj in Pij_hcp_twin[6:12]])))
    return rss_tw_t1, rss_tw_t2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-3: Basic general function related to stress-strain calculation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%3.1 To convert a matrix to voigt array
def symmetric_matrix_to_voigt_array(matrix, format):
    '''
    Input: The symmetric square matrix of n*n dimensions
    Output: vector of the elements from the upper triangular matrix
    Note: If one want to have the Voigt notation otherthan "strain", then
          at type any string otherthan "strain"
    Link: https://en.wikipedia.org/wiki/Voigt_notation
    '''
    matrix = np.array(matrix) # making sure that matrix is an array
    #---------------------------------------------------
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")
    #---------------------------------------------------
    n = matrix.shape[0]
    diagonal = np.diag(matrix) # Extract diagonal elements
    upper_triangular = [] # Extract upper triangular elements (excluding diagonal)
    for i in range(n):
        for j in range(i+1, n):
                if format == "strain":
                    upper_triangular.append(2*matrix[i, j]) ## for strain
                else: 
                    upper_triangular.append(matrix[i, j]) ## for stress
    # Combine diagonal and upper triangular elements
    voigt_vector = np.concatenate([diagonal, upper_triangular[::-1]])
    voigt_matrix = voigt_vector.reshape(-1, 1) # Reshape into a column matrix
    return voigt_matrix

#%% 3.2: To convert voigt-array to matrix
def voigt_array_to_symmetric_matrix(voigt_array, format):
    '''
    Input: Voigt-array, tell forrmat i.e conversion is for "Stress" or "Strain"
    Output: Matrix form of given Voigt notation
    Note: This function does convert the "Voigt" form of stress/strain to
        "Matrix" form
    '''
    # For Stress to convert from voigt to matrix
    n = len(voigt_array) # length of the array
    d = int(n/2) # Dimension of the matrx
    diag_mat = np.zeros([d,d]) #Digonal-part of the matrix
    for i in range(d):
        diag_mat[i,i] = voigt_array[i]
    if format == "strain": ## For strain
        trig_mat = np.zeros([d,d]) #Upper-Trigonal-part of the matrix
        k = d 
        for i in range(d):
            for j in range(i+1,d):
                trig_mat[i,j] = 1/2*(voigt_array[k+2])
                k-=1
        #%% Full-matrix
        sym_mat = diag_mat+trig_mat+trig_mat.T
    else: ## for stress
        trig_mat = np.zeros([d,d]) #Upper-Trigonal-part of the matrix
        k = d 
        for i in range(d):
            for j in range(i+1,d):
                trig_mat[i,j] = voigt_array[k+2]
                k-=1
        #%% Full-matrix
        sym_mat = diag_mat+trig_mat+trig_mat.T 
    return sym_mat

#%% 3.3: To calculate equivalent_stress 
def equivalent_stress(stress_matrix):
    '''
    Input: stress_matrix should be a 3*3 matrix, defined in array form
    Output: Equivalent stress
    Note: It required "dobledot" product function. Also, it is same as if we use the all componets of stress and use standard formula, Mtheod#2
    '''
    ##Volumentric stress component
    sigma11, sigma22, sigma33 = stress_matrix[0,0], stress_matrix[1,1], stress_matrix[2,2]
    ## Devaitoric stress component
    sigma12, sigma13, sigma23 = stress_matrix[0,1], stress_matrix[0,2], stress_matrix[1,2]
    ## Equivalent stress calculation: Method-1
    trace = sigma11+sigma22+sigma33 # Trace of the matrix
    I = np.array([[1,0,0], 
                  [0,1,0],
                  [0,0,1]]) # Identity matrix
    hydrostatic_stress = (1/3)*trace*I # Hydrostaric part of stress state
    deviatoric_stress = stress_matrix-hydrostatic_stress # Deviatoric part of stress-state
    stress_eq = np.sqrt((3/2)*doubledot(deviatoric_stress,deviatoric_stress)) #(3/2) factor ensures that, under uniaxial loading, the von Mises equivalent stress equals the applied stress at yield.
    ## Equivalent stress calculation: Method-2
    # stress_eq = np.sqrt(0.5*(((sigma11-sigma22)**2)+((sigma22-sigma33)**2)+((sigma33-sigma11)**2)+ 6*((sigma12**2)+(sigma13**2)+(sigma23**2))))
    ##%% Note: Both method are same
    return stress_eq

#%% 3.4: To calculate equivalent_strain 
def equivalent_strain(strain_matrix):
    '''
    Input: strain_matrix should be a 3*3 matrix, defined in array form
    Output: Equivalent strain
    Note: strain_eq from both method are different, but finally we are considering Method-1 (FEPX one)
    This function required function "doubledot"
    '''
    e11, e22, e33 = strain_matrix[0,0], strain_matrix[1,1], strain_matrix[2,2]
    e12, e13, e23 = strain_matrix[0,1], strain_matrix[0,2], strain_matrix[1,2]
    ## Method-1: Ref: FEPX
    I = np.array([[1,0,0],
                  [0,1,0], 
                  [0,0,1]])
    trace = e11 + e22 + e33
    hydrostatic = I*(trace)/3
    deviatoric = strain_matrix - hydrostatic
    strain_eq = np.sqrt((2/3)*doubledot(deviatoric,deviatoric))
    # ## Method-2, Accounted Poission ration, Ref: Ansys (https://innovationspace.ansys.com/courses/wp-content/uploads/sites/5/2020/05/Lesson-5-Evaluate-Strain-Results.pdf)
    # nu = 0.33 # poisson's ratio
    # strain_eq_a = (1/(1+nu))*np.sqrt((1/2)*((e11-e22)**2 + (e22-e33)**2 +(e33-e11)**2 + 6*((e12**2) + (e13**2) + (e23**2))))
    # ## Method-3, same as Method#1
    # m  =trace/3
    # strain_eq_j = np.sqrt((2/3)*((e11-m)**2 + (e22-m)**2 +(e33-m)**2 + 2*((e12**2) + (e13**2) + (e23**2))))
    return strain_eq
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-4: Stress calculation based on Hooke's Law, per grain basis
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% 4.1: Stress and other parameters estimation by asssuming "fixed rotation" of grain
def consitutive_behavior_wrt_fix_ori(matchedgrains, orimat, id_n, C_mat):
    '''
    Input: 'matchedgrains': list of matched grains
           'orimat': Rotation tensor in Crystal-to-Sample frame
            'id_n': Grain-ID for which calculation will be done
            C_mat = Stiffness tensor,a  6x6 Matrix
    Output: coo: grains, centroidal
            epln_cf: Strain crystal-frame
            sigma_cf: Stress in crystal-frame
            u_cf: Strain-energy density in crystal-frame
    Note: The output variables can be more calculated, based on choice
    '''
    ## The Centroidal position
    coo = matchedgrains.loc[id_n]["X":"Z"]
    ## Reading strain-tensor, Note: 1000 devison is done to strain values, because strain are in micro-strain in HEDM data
    epln_sf = list_to_matrix_nxn(matchedgrains.loc[id_n]["eKen11":"eKen33"]/1000) # sf = sample frame
    ##----------------------------------Stress calculation---------------------------------------------------
    #%% Converting Strain to Crystal Frame (CF) from Sample frame, to calculate stress in CF
    epln_cf = np.matmul(orimat.T, np.matmul(epln_sf,orimat)) ## cf = crsytal frame
    epln_cf_voigt = symmetric_matrix_to_voigt_array(epln_cf, "strain") # strain in voigt notation with cf
    sigma_cf_voigt = np.matmul(C_mat, epln_cf_voigt) # stress in voigt notation with CF
    sigma_cf = voigt_array_to_symmetric_matrix(sigma_cf_voigt, "stress")
    # sigma_sf = np.matmul(orimat, np.matmul(sigma_cf, orimat.T))
    ##----------------------------------Normlaizing stress and strain wrt corressponding equivalent-stress,and  equivalent-strain ---------------------------------
    epln_cf_eq = equivalent_strain(epln_cf)
    # epln_cf_norm = epln_cf/epln_cf_eq
    sigma_cf_eq = equivalent_stress(sigma_cf)
    # sigma_cf_norm = sigma_cf/sigma_cf_eq
    #----------------------------------Energy calculation-----------------------------------------------------------
    u_cf = 0.5*np.matmul(sigma_cf_voigt.flatten(), epln_cf_voigt.flatten().T) # .flatten() is added to make 2D array to single array
    return coo, epln_cf, epln_cf_eq, sigma_cf, sigma_cf_eq, u_cf

def consitutive_behavior(matchedgrains, id_n, C_mat):
    '''
    Input: 'matchedgrains': list of matched grains
            'id_n': Grain-ID for which calculation will be done
            C_mat = Stiffness tensor,a  6x6 Matrix
    Output: coo: grain'centroidal,
            ori: grain' orientation,
            epln_cf: Strain crystal-frame,
            sigma_cf: Stress in crystal-frame,
            u_cf: Strain-energy density in crystal-frame,
    Note: The output variables can be more calculated, based on choice
    '''
    coo = matchedgrains.loc[id_n]["X":"Z"]
    orimat = list_to_matrix_nxn(matchedgrains.loc[id_n]["O11":"O33"]) ## Crystal to sample, and ori_parent.T meaning sample to crystal 
    epln_sf = list_to_matrix_nxn(matchedgrains.loc[id_n]["eKen11":"eKen33"]/1000) # sf = sample frame
    ## Reading strain-tensor, Note: 1000 devison is done to strain values, because strain are in micro-strain in HEDM data
    #----------------------------------Stress calculation-----------------------------------------------------------
    #%% Converting Strain to Crystal Frame from Sample frame, to calculate stress in CF
    epln_cf = np.matmul(orimat.T, np.matmul(epln_sf,orimat)) ## cf = crsytal frame
    epln_cf_voigt = symmetric_matrix_to_voigt_array(epln_cf, "strain") # strain in voigt notation with cf
    sigma_cf_voigt = np.matmul(C_mat, epln_cf_voigt) # stress in voigt notation with CF
    sigma_cf = voigt_array_to_symmetric_matrix(sigma_cf_voigt, "stress")
    # sigma_sf = np.matmul(orimat, np.matmul(sigma_cf, orimat.T))
    ##**************************Normlaizing stress and strain wrt corressponding equivalent-stress,and  equivalent-strain *********************
    epln_cf_eq = equivalent_strain(epln_cf)
    # epln_cf_norm = epln_cf/epln_cf_eq
    sigma_cf_eq = equivalent_stress(sigma_cf)
    # sigma_cf_norm = sigma_cf/sigma_cf_eq
    #----------------------------------Energy calculation-----------------------------------------------------------
    u_cf = 0.5*np.matmul(sigma_cf_voigt.flatten(), epln_cf_voigt.flatten().T) # .flatten() is added to make 2D array to single array
    return coo, orimat, epln_cf, epln_cf_eq, sigma_cf, sigma_cf_eq, u_cf

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Part-4: RSS estimation from using Schmid-tensor 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def StressStrainEnergyRSS_loadwise_fixori(pth, sam_n, load_range, twin_load,ls_n, gr_n,fix_ori,sorted_IDs,C_tensor, schmid_tensor, deformation_type):
    '''
    Input: "load_range" = list of applied loads; for parent-grains it will be from starting i.e. load-0, for "twin" it shouild be from the load at which twin form
            "Twin_load" = Load at which twin formed or we are considereing twinning
           "ls" = a load-step("ls") from which one either want to analyze 'slip' or 'twin' analysis; for slip i.e. for parent it will always be "ls=0", for twin it will be "ls" at twin form
           "gr_n" = grain number or which grain? 
           "fix_ori" = Orientation wrt that we want to do the analysis, mostly it is of the laod-step 0 (default) but could be from any load-step
           "sorted_IDs" = sorted list of grains-IDs for which we want to these calculations
           "C_tensor" = Stiffness tensor, i.e. a 6*6 matrix
           "Schmid_tensor" = Schmid-tensor either for slip-systems or twin-systems
           "deformation_type" = for slip-systems type "all_slip" and for tensile-twin type "tensile_twins"
    Output: "cord" = cordinates of the grains/twins
            "strain" = estimated strain in crystal frame
            "stress" = estimated stress in crystal frame
            "rss_basal" = estimated RSS value for basal slip similarly other
    Note: 1. All outputs will be in dictionary formmat with key being "ls_{ls}", where {ls} is stand for load-steps 
         2. Till in this analysis, deformation due to 'compresssion' twin are not considered
         3. Based on the choice or requirment, one can calculated the other properties too, such as equivalent stress-strain, etc
         4. Make sure that the following functions are in the same script or library where this function is stored-
             i)- consitutive_behavior_wrt_fix_ori
            ii)- rss_hcp_all_slip_a_grain/rss_hcp_all_tensiletwin_a_grain
    '''
    strain, strain_eq = {}, {}
    stress, stress_eq = {}, {}
    cord, u = {}, {}
    rss_sl_basal,rss_sl_prism,rss_sl_pyramI_a,rss_sl_pyramI_ca = {},{},{},{} # Dictionaries for all slip-systems: Basal, Prismatic< Pyramidal
    rss_tw_t1, rss_tw_t2 = {}, {} # Dictionaries for all tensile-systems: T1(85 deg misori), T2(35 deg.misori)
    for ii,load_c in enumerate(load_range, start=ls_n):
        ## Selecting grain IDs
        # grain_ids = sorted_IDs[f"ls_{ii}"][pg_tw_pair_n]
        #%%.......................................................................................
        if deformation_type == "all_slip": 
            grain_ids = sorted_IDs[f"load_{twin_load}_gr_{gr_n}"][ii] #"gr" = parent or neighbor grain, slip based deformation, saved as dictionary
        elif deformation_type == "tensile_twins":
            grain_ids = sorted_IDs[f"load_{twin_load}_tw_{gr_n}"][ii] # tw = twinning
        #%%.......................................................................................
        df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
        # ##***************************************************** PARENT GRAIN ANALYSIS******************************************************************    
        ### Parent-Grain constitutive behavior 
        coo, epln_cf, epln_cf_eq, sigma_cf, sigma_cf_eq, u_cf = consitutive_behavior_wrt_fix_ori(df_final_matchgrains,fix_ori,grain_ids, C_tensor)
        #-------------------------------------------------
        cord[f"ls_{ii}"] = coo
        strain[f"ls_{ii}"] = epln_cf
        strain_eq[f"ls_{ii}"] = epln_cf_eq
        stress[f"ls_{ii}"] = sigma_cf
        stress_eq[f"ls_{ii}"] = sigma_cf_eq
        u[f"ls_{ii}"] = u_cf
        ### Grains RSS analysis 
        if deformation_type == "all_slip":
            Pij_hcp_slip = schmid_tensor
            rss_basal,rss_prism,rss_pyramI_a, rss_pyramI_ca = rss_hcp_all_slip_a_grain(Pij_hcp_slip, sigma_cf)
            rss_sl_basal[f"ls_{ii}"] = rss_basal
            rss_sl_prism[f"ls_{ii}"] = rss_prism
            rss_sl_pyramI_a[f"ls_{ii}"] = rss_pyramI_a
            rss_sl_pyramI_ca[f"ls_{ii}"] = rss_pyramI_ca
        elif deformation_type == "tensile_twins":
            Pij_hcp_twin = schmid_tensor
            rss_twin_t1,rss_twin_t2 = rss_hcp_all_tensiletwin_a_grain(Pij_hcp_twin, sigma_cf)
            rss_tw_t1[f"ls_{ii}"] = rss_twin_t1
            rss_tw_t2[f"ls_{ii}"] = rss_twin_t2
    #%% Retuening functions
    if deformation_type == "all_slip":
        return cord,strain,stress,u,rss_sl_basal,rss_sl_prism,rss_sl_pyramI_a,rss_sl_pyramI_ca
    elif deformation_type == "tensile_twins":
        return cord,strain,stress,u,rss_tw_t1, rss_tw_t2
    
def StressStrainEnergyRSS_loadwise(pth, sam_n, load_range, twin_load,ls_n, gr_n,sorted_IDs,C_tensor, schmid_tensor, deformation_type):
    '''
    Input: "load_range" = list of applied loads; for parent-grains it will be from starting i.e. load-0, for "twin" it shouild be from the load at which twin form
           "ls" = a load-step("ls") from which one either want to analyze 'slip' or 'twin' analysis; for slip i.e. for parent it will always be "ls=0", for twin it will be "ls" at twin form
           "gr_n" = grain number or which grain want to be analyzed?
           "fix_ori" = Orientation wrt that we want to do the analysis, mostly it is of the laod-step 0 (default) but could be from any load-step
           "sorted_IDs" = sorted list of grains-IDs for which we want to these calculations
           "C_tensor" = Stiffness tensor, i.e. a 6*6 matrix
           "Schmid_tensor" = Schmid-tensor either for slip-systems or twin-systems
           "deformation_type" = for slip-systems type "all_slip" and for tensile-twin type "tensile_twins"
    Output: "cord" = cordinates of the grains/twins
            "strain" = estimated strain in crystal frame
            "stress" = estimated stress in crystal frame
            "rss_basal" = estimated RSS value for basal slip similarly other
    Note: 1. All outputs will be in dictionary formmat with key being "ls_{ls}", where {ls} is stand for load-steps 
         2. Till in this analysis, deformation due to 'compresssion' twin are not considered
         3. Based on the choice or requirment, one can calculated the other properties too, such as equivalent stress-strain, etc
         4. Make sure that the following functions are in the same script or library where this function is stored-
             i)- consitutive_behavior_wrt_fix_ori
            ii)- rss_hcp_all_slip_a_grain/rss_hcp_all_tensiletwin_a_grain
    '''
    strain, strain_eq = {}, {}
    stress, stress_eq = {}, {}
    cord,ori, u = {}, {},{}
    rss_sl_basal,rss_sl_prism,rss_sl_pyramI_a,rss_sl_pyramI_ca = {},{},{},{} # Dictionaries for all slip-systems: Basal, Prismatic< Pyramidal
    rss_tw_t1, rss_tw_t2 = {}, {} # Dictionaries for all tensile-systems: T1(85 deg misori), T2(35 deg.misori)
    for ii,load_c in enumerate(load_range, start=ls_n):
        ## Selecting grain IDs
        # grain_ids = sorted_IDs[f"ls_{ii}"][pg_tw_pair_n]
        #%%.......................................................................................
        if deformation_type == "all_slip": 
            grain_ids = sorted_IDs[f"load_{twin_load}_gr_{gr_n}"][ii] #"gr" = parent or neighbor grain, slip based deformation, saved as dictionary
        elif deformation_type == "tensile_twins":
            grain_ids = sorted_IDs[f"load_{twin_load}_tw_{gr_n}"][ii] #"tw" = twin based deformation, saved as dictionary
        #%%.......................................................................................
        df_final_matchgrains = pd.read_csv(f"{pth}/sample_{sam_n}/csv_files/sam{sam_n}_{load_c}N_grains.csv", skiprows=1, sep = "\t").set_index(["%new_GrainID"])
        # ##***************************************************** PARENT GRAIN ANALYSIS******************************************************************    
        ### Parent-Grain constitutive behavior 
        coo, orimat, epln_cf, epln_cf_eq, sigma_cf, sigma_cf_eq, u_cf = consitutive_behavior(df_final_matchgrains,grain_ids, C_tensor)
        #-------------------------------------------------
        cord[f"ls_{ii}"] = coo
        ori[f"ls_{ii}"] = orimat
        strain[f"ls_{ii}"] = epln_cf
        strain_eq[f"ls_{ii}"] = epln_cf_eq
        stress[f"ls_{ii}"] = sigma_cf
        stress_eq[f"ls_{ii}"] = sigma_cf_eq
        u[f"ls_{ii}"] = u_cf
        ### Grains RSS analysis 
        if deformation_type == "all_slip":
            Pij_hcp_slip = schmid_tensor
            rss_basal,rss_prism,rss_pyramI_a, rss_pyramI_ca = rss_hcp_all_slip_a_grain(Pij_hcp_slip, sigma_cf)
            rss_sl_basal[f"ls_{ii}"] = rss_basal
            rss_sl_prism[f"ls_{ii}"] = rss_prism
            rss_sl_pyramI_a[f"ls_{ii}"] = rss_pyramI_a
            rss_sl_pyramI_ca[f"ls_{ii}"] = rss_pyramI_ca
        elif deformation_type == "tensile_twins":
            Pij_hcp_twin = schmid_tensor
            rss_twin_t1,rss_twin_t2 = rss_hcp_all_tensiletwin_a_grain(Pij_hcp_twin, sigma_cf)
            rss_tw_t1[f"ls_{ii}"] = rss_twin_t1
            rss_tw_t2[f"ls_{ii}"] = rss_twin_t2
    #%% Retuening functions
    if deformation_type == "all_slip":
        return cord,ori,strain,stress,u,rss_sl_basal,rss_sl_prism,rss_sl_pyramI_a,rss_sl_pyramI_ca
    elif deformation_type == "tensile_twins":
        return cord,ori,strain,stress,u,rss_tw_t1, rss_tw_t2
    
#%% Using by class-
class StressStrainEnergyRSS:
    '''
    Description: This class is to compute Stress, Strain, Energy and RSS load-step-wise 
                 for slip or twin deformation in HCP crystals 
    '''
    #%%===========================================================
    #%% Initialization
    def __init__(self,pth, sam_n,C_tensor,schmid_tensor):
        self.pth = pth
        self.sam_n = sam_n
        self.C_tensor = C_tensor
        self.schmid_tensor = schmid_tensor
    #%%===========================================================
    #%% Main Function 
    def compute(self, load_range, ls_n, gr_n, fix_ori,sorted_IDs, deformation_type):
        # ---- Output Dictionaries ----
        cord,u = {},{}
        strain, strain_eq = {}, {}
        stress, stress_eq = {}, {}
        # Slip RSS
        rss_pg_basal,rss_pg_prism,rss_pg_pyramI_a,rss_pg_pyramI_ca = {},{},{},{}
        # Twin RSS
        rss_tw_t1,rss_tw_t2 = {},{}
        #%%===========================================================
        #%% Looping over load-steps
        for ii, load_c in enumerate(load_range, start=ls_n):
            # ------ Grain IDs for this load step ------
            grain_ids = sorted_IDs[f"LS_{ii}"][gr_n]
            # ------ Load CSV ------
            df = pd.read_csv(f"{self.pth}/sample_{self.sam_n}/csv_files/sam{self.sam_n}_{load_c}N_grains.csv",skiprows=1, sep="\t").set_index(["%new_GrainID"])
            #%% =================================================================
            #%% PARENT-GRAIN CONSTITUTIVE RESPONSE
            coo, epln_cf, epln_cf_eq,sigma_cf, sigma_cf_eq, u_cf = consitutive_behavior_wrt_fix_ori(df, fix_ori, grain_ids, self.C_tensor)
              # -------- Store results --------
            ls_key = f"ls_{ii}"
            cord[ls_key] = coo
            strain[ls_key] = epln_cf
            strain_eq[ls_key] = epln_cf_eq
            stress[ls_key] = sigma_cf
            stress_eq[ls_key] = sigma_cf_eq
            u[ls_key] = u_cf
            #%% =================================================================
            #%% RSS BASED ON deformation_type
            if deformation_type == "all_slip":
                rss_basal, rss_prism,rss_pyramI_a, rss_pyramI_ca = rss_hcp_all_slip_a_grain(self.schmid_tensor, sigma_cf)
                rss_pg_basal[ls_key] = rss_basal
                rss_pg_prism[ls_key] = rss_prism
                rss_pg_pyramI_a[ls_key] = rss_pyramI_a
                rss_pg_pyramI_ca[ls_key] = rss_pyramI_ca
            elif deformation_type == "tensile_twins":
                rss_t1, rss_t2 = rss_hcp_all_tensiletwin_a_grain(self.schmid_tensor, sigma_cf)
                rss_tw_t1[ls_key] = rss_t1
                rss_tw_t2[ls_key] = rss_t2
        # =====================================================================
        #  RETURN BASED ON deformation_type
        if deformation_type == "all_slip":
            return cord, strain, stress,rss_pg_basal, rss_pg_prism,rss_pg_pyramI_a, rss_pg_pyramI_ca
        elif deformation_type == "tensile_twins":
            return cord, strain, stress,rss_tw_t1, rss_tw_t2

