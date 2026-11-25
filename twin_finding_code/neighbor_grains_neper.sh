#!/bin/bash
: '
Date: 8th April 2025
Author: Amit, ACDM Lab, Tuscaloosa, AL, USA. Email: amitdravid@outlook.com, asingh38@crimson.ua.edu
Description: This Script is to find out the neighboring grains to the parent-grains which got twinned
Neper-version: 4.10.2-160-devel, used
Inputs: 1. One needs the dimensions of measured domain volume from HEDM, to finilize the volume in Neper
Notes: 
1. This script require: (i) The measured domain dimesions, (ii) The centroid, radius, diameters and orientations of the measured grains in seprate file, 
   (iii) centroid-diameter of grains in combined file. These could be find by running "neighbr_grains_stat.py". 
2. The "neighbr_grains_stat.py" is kind of intermediate script and use as pre-post supplementry file to this. 
3. Care must be taken on the input file names convetion used here. if required change the input file name, in the script based on save files.
4. "-loadtess" command only applicable while using Neper -T not on Neper -V.
5. This script asks two types of load inputs: 'load_tess' and 'load_twin' at which tesselation is being generated, and twin form respectively. If the tessellations 
        are generated for each load in which twin might have formed then that case load_tess and load_twin could be same. 
        However, it is very obvious too that the tesselation (micorstrcure) should not change much especially with in elatsic-limit of load. Therefore, one can assume the same, 
        tessellation across all load steps. As a result one could use the same tesselaltion generated for one load. In the present study I am using tessellation generated at load 155N.
'
#***************************************************************
#Part-0: Input variables and defining path 
#***************************************************************
read -p "Enter sample no.:" sam_n
read -p "Enter load of tesselation:" load_c
pth=/home/asingh38/projects/nsf_twinning_2022/raw_data/sample_${sam_n}/neighbour_grains

#**************************************************************************************************************
#%% Part-1: Generating Domain 
#**************************************************************************************************************
#%% Note: 1. To generate domain, one has to prior run the "neighbr_grains_stat.py"
#%% 2. Careful with translating the domain based on dimensions of the measured volume, one may need to play with "translate"
#**************************************************************************************************************
#%% 1.1: Constructing Neper 'domain' based on HEDM measured volume, to make sure both measured and Neper generated domain should be same
: '
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Caution:Dimensions of the domian shoud be enter wrt unit length"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
read -p "Enter HEDM measured volume dimensions(l w h, space seprated):" l w h

#%% 1.1.1: Tesselations
neper -T -n 1 \
        -dim "3" \
        -domain "cube($l,$w,$h):translate(-0.5,-0.5,-0.225)" \
        -o ${pth}/domain_new_${load_c}
'
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% 1.1.2: Domain visualization, just to check. Note: It required 'centroidal and diameters of the grains measured from HEDM. Agian check the values, 
#%%  Neper only uses units from 0-1, while HEDM units are in micro so, one may need to divide them with 1000, before using them to generated tessellations
: '
neper -V ${pth}/domain_new_${load_c}.tess,${pth}/centroidals_load${load_c}.txt \
        -crysym hexagonal \
        -datapointcol "ori:file(${pth}/orientations_load${load_c}.txt, des=rotmat)"\
        -datapointrad "file(${pth}/diameters_load${load_c}.txt)"\
        -datapointtrs 0.5 \
        -datacelltrs 1 \
        -dataedgerad 0.005 \
        -cameraangle 16.5 \
        -datapointcolscheme "ipf" \
        -print ${pth}/allgrains_centroids_neper_load${load_c}_size_new

 '
#**************************************************************************************************************
##% Part-2: Generating and visualization Tessellations
##% Note: At present, script only handle one type of tessellation generation using sphereicity. It could be looped over for many different types of tessellations
#**************************************************************************************************************
##% 2.1: Input variables
read -p "Enter the mean, and std of grain'sphericity(space seprated):" mu sigma
n_grains=$(wc -l < "${pth}/centroidals_load${load_c}.txt")
echo "Numberof grains:${n_grains}"
wt_method="rad"
tess_file="${pth}/tess/nghbrs_centdia_mu${mu}_sig${sigma}_w${wt_method}_load${load_c}"
##% 2.2 Tessellation generation
: '
neper -T -n "${n_grains}" \
        -dim "3" \
        -domain "cube($l,$w,$h):translate(-0.5,-0.5,-0.225)" \
        -morpho "centroiddiameq:file(${pth}/centroidals_dia_load${load_c}.txt),1-sphericity:lognormal(${mu},${sigma})" \
        -morphooptiini "coo:file(${pth}/centroidals_load${load_c}.txt),weight:file(${pth}/radius_load${load_c}.txt)" \
        -morphooptistop "eps<1e-3" \
        -crysym "hexagonal" \
        -ori "file(${pth}/orientations_load${load_c}.txt,des=rotmat)" \
        -regularization "0" \
        -o "${tess_file}"
'
##%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##%2.3 Tessellation Visualization
: '
neper -V ${tess_file}.tess \
        -datacellcolscheme "ipf" \
        -datacelltrs "0.3" \
        -dataedgerad "0.002" \
        -cameraangle "16.5" \
        -print ${tess_file}
'
#**************************************************************************************************************
##% Part-3: Statistical analysis of neper generated grains:: Extracting centroidal (coordinates), diameters and number of neighbor grains
#%% Note: Care must be taken on the 'output' file name genrated with "statcell" command, because Neper does not save the file with  numeric number
#**************************************************************************************************************
out_file=nghbrs_centdia_mu${mu}_sig${sigma}_load${load_c}
##% 3.1 Extracting the "centroidal(coordinates)" of the grains/cell
: '
neper -T -loadtess ${tess_file}.tess \
        -statcell coo \
        -o ${pth}/tess/stcell/coo_${out_file}
'
#%Warning:: Please check the output file name, and edit it manually, else it will overwrite on the previous file
##%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##% 3.2 Extracting the "equivalent-diameter" of the grains/cell i.e diameq
: '
neper -T -loadtess ${tess_file}.tess \
        -statcell diameq \
        -o ${pth}/tess/stcell/diameq_${out_file}
'
##%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##% 3.3 Extracting the "Number of neighboring grains" per grain or cell/poly
: '
neper -T -loadtess ${tess_file}.tess \
        -statpoly npolys \
        -o ${pth}/tess/stcell/npolys_${out_file}
'
#*****************************************************************************************************************************************************************
##% Part-4: Visulaization of parent and neighbor-grains or choice of grains using boolean in Neper
#%% Note: The boolean file is required to visualize the grains/cell or poly using Neper. Indeed, the boolean for the selected grain/s should be 1.
#%% Yes length of variables in boolean file should be same as number of grains.
#********************************************************************************************************************************************************************
#%% 4.1 Visulization of parent grain only
read -p "Enter misori angle:" twin_angle
read -p "Enter twin load:" load_twin
read -p "Enter parent-twin pair number:" pg_tw_pair_n
# bool_file=parent_bool_misori${twin_angle}_load${load_c}
bool_file=pg_pair${pg_tw_pair_n}_bool_misori${twin_angle}_twload${load_twin}
/usr/local/bin/neper -V ${tess_file}.tess \
        -crysym hexagonal \
        -datacellcol ori \
        -datacellcolscheme ipf \
        -datacelltrs "0.3" \
        -showpoly "file(${pth}/tess/stcell/parent/${bool_file}.txt)" \
        -showedge "poly_shown||domtype==1" \
        -cameraangle 16.5 \
        -print ${pth}/tess/stcell/parent/${bool_file}

        # -showcsys  "1" \
        # -datacsyslabel "X:Y:Z" \
        # -cameracoo "1.5:-1.5:2" \
        
# #%% 4.2 Visulization of parent-neighbor grains
bool_file=nghbr_bool_pg_pair${pg_tw_pair_n}_misori${twin_angle}_twload${load_twin}_w${wt_method}
/usr/local/bin/neper -V ${tess_file}.tess \
        -crysym hexagonal \
        -datacellcol ori \
        -datacellcolscheme ipf \
        -datacelltrs "0.3" \
        -showpoly "file(${pth}/tess/stcell/neighbors/${bool_file}.txt)" \
        -showedge "poly_shown||domtype==1" \
        -cameraangle 16.5 \
        -print ${pth}/tess/stcell/neighbors/${bool_file}








