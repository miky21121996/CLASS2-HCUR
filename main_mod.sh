#!/bin/bash -l

source /work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/current_val.sh

#bsub -K -n 1 -q s_long -J EANCALC -e aderr_0 -o adout_0 -P 0510 "python obs_extraction.py $date_in $date_fin $path_to_metadata_obs_file $time_res $path_to_out_obs_ts $depth_obs $nan_treshold $path_to_accepted_metadata_obs_file" &

#wait

IFS=',' read -r -a name_exp_array <<< "$name_exp"

bsub -K -n 1 -q s_long -J EANCALC -e aderr_1 -o adout_1 -P 0510 "python nearest_mod.py $time_res $date_in $date_fin $path_to_mod_output $path_to_accepted_metadata_obs_file $path_to_mask_file $path_to_destag_output_folder $path_to_out_mod_ts $depth_obs ${name_exp_array[@]}" &
