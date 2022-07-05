#!/bin/bash -l

#source /work/oda/mg28621/prova_destag/current_val.sh
source /work/oda/mg28621/prova_destag/surface_insitu_validation/current_val_plot.sh
#python prova_destag.py 20201001 20201130 /work/oda/ag15419/exp/eas6_v8/simu_ctrl0/output/ /work/oda/mg28621/prova_destag/ simu_ctrl0 1d /work/oda/mg28621/prova_destag/mesh_mask.nc

#python destaggering.py $date_in $date_fin $path_to_mod_output $path_to_destag_output_folder $name_exp $time_res $path_to_mask_file

#echo "name_exp: $name_exp"
#IFS=',' read -r -a name_exp_array <<< "$name_exp"
#echo "name_exp_array: ${name_exp_array[@]}"
#num_exp=${#name_exp_array[@]}

IFS=',' read -r -a path_to_out_mod_ts_array <<< "$path_to_out_mod_ts"
echo "path_to_out_mod_ts_array: ${path_to_out_mod_ts_array[@]}"

bsub -K -n 1 -q s_long -J EANCALC -e aderr_1 -o adout_1 -P 0510 "python plot_curr_val.py $path_to_obs_file $path_to_out_obs_ts $path_to_output_plot_folder $date_in $date_fin $time_res $time_res_xaxis $num_exp ${path_to_out_mod_ts_array[@]}"
