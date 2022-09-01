#!/bin/sh
#
# by Michele Giurato (CMCC)
# michele.giurato@cmcc.it
#
# Written: 31/03/2022
# Last Mod: 31/03/2022
#
# Ini file for velocity current validation
#
####################### SET THE FOLLOWING VARS: ############################################

# ------ Input variables---------

#[My Section]

# INPUT VARIABLES

date_in=20200101
date_fin=20211230
path_to_mod_output=/work/oda/med_dev/EAS7/output/
name_exp=mfs1,mfs2
time_res=1d
path_to_mask_file=/work/oda/mg28621/prova_destag/surface_insitu_validation/mesh_mask.nc
path_to_metadata_obs_file=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/cmems_moorings.csv
path_to_accepted_metadata_obs_file=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/prova_obs_metadata_accepted_2020_2021_treshold_0p5_eas7.csv
depth_obs=3
nan_treshold=0.5

# OUTPUT VARIABLES
path_to_destag_output_folder=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/destaggered_UV_folder_2020_2021_eas7/
path_to_out_mod_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/prova_output_mod_ts_2020_2021_3m_completed_treshold_0p5_eas7/
path_to_out_obs_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/prova_output_obs_ts_2020_2021_3m_completed_treshold_0p5_eas7_correct_vida/
