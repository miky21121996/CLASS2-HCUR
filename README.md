# surface_curr_val

1. Open current_val.sh
  a. date_in: initial date  
  b. date_fin: final date  
  c. path_to_mod_output: location of model files to be used  
  d. name_exp: name of the model experiment (that is specified in the name of the model files)  
  e. time_res: time resolution of model files (1d, 1h...)  
  f. path_to_mask_file  
  g. path_to_metadata_obs_file  
  h. path_to_accepted_metadata_obs_file: location of metadata file created after checks  
  i. path_to_destag_output_folder: location of folder in which destaggered model files are created  
  l. path_to_out_mod_ts: location of extracted model time series nc files  
  m. path_to_out_obs_ts: location of extracted observation time series nc files  
  n. path_to_output_plot_folder: location of folder in which plots are created  
  o. depth_obs: depth around which model values are extracted to compare (leave 3)  
  p. nan_treshold: percentage treshold above which values are not accepted  
    
2. run: sh main_mod.sh  
3. Open current_val_plot.sh  
  a. date_in  
  b. date_fin  
  c. num_exp: number of experiments. If num_exp = 1 main_plot.sh will provide plots about validation of the single experiment against observations. If num_exp > 1, main_plot.sh will provide plots about comparison between the experiments.  
  d. time_res  
  e. path_to_obs_file: location of metadata file created after checks  
  f. path_to_out_mod_ts: location of extracted model time series nc files, name you want to give in the plots  
  Example: if num_exp = 1, path_to_out_mod_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/output_mod_ts_2019_3m_completed_treshold_0p4_eas5_hp_v2/,EAS5  
           if num_exp = 2, path_to_out_mod_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/output_mod_ts_2019_3m_completed_treshold_0p4_eas5_hp_v2/,/work/oda/mg28621/prova_destag/surface_insitu_validation/output_mod_ts_2019_3m_completed_treshold_0p4_minr_ctrl0/,EAS5,minr_ctrl0  
  g. path_to_out_obs_ts: location of extracted observation time series nc files  
  h. path_to_output_plot_folder: location of folder in which plots are created  
  i. time_res_xaxis: resolution of x axis in the plots (example-> 2w: each two weeks)  
4. run: sh main_plot.sh  
