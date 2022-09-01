#!/bin/bash -l


source /work/oda/mg28621/prova_destag/surface_insitu_validation/surface_insitu_validation/current_val_plot.sh

IFS=',' read -r -a path_to_out_mod_ts_array <<< "$path_to_out_mod_ts"
echo "path_to_out_mod_ts_array: ${path_to_out_mod_ts_array[@]}"

bsub -K -n 1 -q s_long -J EANCALC -e aderr_1 -o adout_1 -P 0510 "python plot_curr_val.py $path_to_obs_file $path_to_out_obs_ts $path_to_output_plot_folder $date_in $date_fin $time_res $time_res_xaxis $num_exp ${path_to_out_mod_ts_array[@]}"
