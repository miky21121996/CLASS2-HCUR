# CLASS2-HCUR-MO
This tool provides statistics comparing model data and mooring surface current velocities.

## Table of Contents
* [General Info](#general-info)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Contact](#contact)

## General Info
The tool can manage the comparison of multiple experiments with the same observation dataset.

## Setup
It is necessary to have a python virtual environment with the following packages:  
* sys  
* warnings  
* netCDF4  
* datetime  
* os  
* math
* csv  
* numpy  
* xarray (version 0.20.1)
* scipy  
* matplotlib  
* mpl_toolkits.basemap (version 1.2.2)  
* statistics  
* statsmodels

To clone the repository: git clone git@github.com:miky21121996/CLASS2-HCUR-MO.git name_your_repo  

## Usage

1. Open *current_val.ini*

**INPUT VARIABLES**:

* date_in: initial date  
* date_fin: final date  
* path_to_mod_output: location of model files to be used  
* name_exp: name of the model experiment (that is specified in the name of the model files). If the model files of the experiment change name over time, write all the names separated by comma (ex: mfs1,mfs2)  
* time_res: time resolution of model files (1d, 1h...). You can only average the model files daily  
* path_to_mask_file: location of mask file associated to the model outputs  
* path_to_metadata_obs_file  
* depth_obs: depth around which model values are extracted to compare (leave 3)  
* nan_treshold: percentage treshold above which obs and mod values are not accepted
  
 **OUTPUT VARIABLES**
 
* work_dir: directory where all tool will work
* path_to_accepted_metadata_obs_file: location of metadata file created after checks
* path_to_destag_output_folder: location of folder in which destaggered model files are created (if you want to create destaggered files from scratch, delete the folder before the run, if you already have destaggered files just indicate the path) 
* path_to_out_mod_ts: location of extracted model time series nc files  
* path_to_out_obs_ts: location of extracted observation time series nc files  
     
2. run sh main_mod_obs_extraction.sh &  

3. Open *current_val_plot.ini*  

**INPUT VARIABLES**:

* date_in  
* date_fin  
* num_exp: number of experiments. If num_exp = 1 main_plot.sh will provide plots about validation of the single experiment against observations. If num_exp > 1, main_plot.sh will provide plots about comparison between the experiments (so, the first step is to run sh main_mod_obs_extraction.sh for each experiment).  
* time_res  
* path_to_obs_file: location of metadata file created after checks done executing main_mod_obs_extraction.sh  
* path_to_out_mod_ts: location of extracted model time series nc files + name of experiment you want to be shown in the plots  
  Example: if num_exp = 1, path_to_out_mod_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/output_mod_ts_2019_3m_completed_treshold_0p4_eas5_hp_v2/,EAS5  
           if num_exp = 2, path_to_out_mod_ts=/work/oda/mg28621/prova_destag/surface_insitu_validation/output_mod_ts_2019_3m_completed_treshold_0p4_eas5_hp_v2/,/work/oda/mg28621/prova_destag/surface_insitu_validation/output_mod_ts_2019_3m_completed_treshold_0p4_minr_ctrl0/,EAS5,minr_ctrl0  
* path_to_out_obs_ts: location of extracted observation time series nc files
* time_res_xaxis: resolution of x axis in the plots (example-> 2w: each two weeks)
  
**OUTPUT VARIABLES**

* work_plot_dir
* path_to_output_plot_folder: location of folder in which plots are created  

4. run sh main_plot.sh &  

## Project Status
Project is: _in progress_ 

## Contact
Created by Michele Giurato (michele.giurato@cmcc.it) - feel free to contact me!
