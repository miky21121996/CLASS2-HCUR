#!/bin/bash -l

source /work/oda/mg28621/prova_destag/surface_insitu_validation/current_val.sh

#python prova_destag.py 20201001 20201130 /work/oda/ag15419/exp/eas6_v8/simu_ctrl0/output/ /work/oda/mg28621/prova_destag/ simu_ctrl0 1d /work/oda/mg28621/prova_destag/mesh_mask.nc

#python destaggering.py $date_in $date_fin $path_to_mod_output $path_to_destag_output_folder $name_exp $time_res $path_to_mask_file
#bsub -K -n 1 -q s_long -J EANCALC -e aderr_0 -o adout_0 -P 0510 "python obs_extraction.py $date_in $date_fin $path_to_metadata_obs_file $time_res $path_to_out_obs_ts $depth_obs $nan_treshold $path_to_accepted_metadata_obs_file" &

#wait

bsub -K -n 1 -q s_long -J EANCALC -e aderr_1 -o adout_1 -P 0510 "python nearest_mod.py $name_exp $time_res $date_in $date_fin $path_to_mod_output $path_to_accepted_metadata_obs_file $path_to_mask_file $path_to_destag_output_folder $path_to_out_mod_ts $depth_obs" &


#python plot_curr_val.py $path_to_out_mod_ts $path_to_out_mod_ts $path_to_output_plot_folder $date_in $date_fin $time_res

