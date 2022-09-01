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

#INPUT VARIABLES
date_in=20200101
date_fin=20211230
num_exp=1
time_res=1d
path_to_obs_file=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/prova_obs_metadata_accepted_2020_2021_treshold_0p5_eas7.csv
path_to_out_mod_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/prova_output_mod_ts_2020_2021_3m_completed_treshold_0p5_eas7/,EAS7
path_to_out_obs_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/prova_output_obs_ts_2020_2021_3m_completed_treshold_0p5_eas7_correct_vida/
time_res_xaxis=1m

#OUTPUT VARIABLES
path_to_output_plot_folder=/work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/prova_output_plot_2020_2021_3m_completed_treshold_0p5_eas7_no_mykon_correct_vida/
