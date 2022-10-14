import warnings
warnings.filterwarnings("ignore")
import sys
import netCDF4 as nc4
import os.path
from os import listdir
from os.path import isfile, join
import math
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from datetime import timedelta, date
import csv
from destaggering_UV import destaggering

def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def find_nearest_mod_point(obs_file, end_date, name_exp, temp_res, path_to_destag_output_folder, depth_obs, t_mask):
    nearest_point = {}    
    for name, obs in obs_file.items():

        print(name)
        timetag = end_date.strftime("%Y%m%d")
        counter=0
        for exp_name in name_exp:
            destaggered_u_file= exp_name + "_" + temp_res + "_" + timetag + "_grid_U2T.nc"
            #destaggered_v_file= exp_name + "_" + temp_res + "_" + timetag + "_grid_V2T.nc"
            try:
                U_current = xr.open_dataset(path_to_destag_output_folder + destaggered_u_file)
                #V_current = xr.open_dataset(path_to_destag_output_folder + destaggered_v_file)
            except Exception:
                counter=counter+1
                continue
        if counter==len(name_exp):
           continue

        vettore_lat=U_current.nav_lat.data[:,0]
        vettore_lon=U_current.nav_lon.data[0,:]
        vettore_depth=U_current.deptht.data

        depth_osservazione = float(obs[depth_obs])
        array_depth = np.column_stack(U_current.deptht.values)
        nemo_tree = cKDTree(array_depth.T)
        dist_depth, idx_near_obs_depth = nemo_tree.query(np.array([depth_osservazione]))

        masked_nav_lat = U_current.nav_lat.values * np.squeeze(t_mask[idx_near_obs_depth,:,:])
        masked_nav_lon = U_current.nav_lon.values * np.squeeze(t_mask[idx_near_obs_depth,:,:])
        masked_nav_lat[masked_nav_lat==0] = np.nan
        masked_nav_lon[masked_nav_lon==0] = np.nan

        nemo_lonlat = np.column_stack((masked_nav_lat.ravel(), masked_nav_lon.ravel()))
        nemo_lonlat = nemo_lonlat[~np.isnan(nemo_lonlat).any(axis=1)]
        nemo_tree = cKDTree(nemo_lonlat)
        dist, idx_near_obs = nemo_tree.query(np.column_stack((float(obs[last_lat]), float(obs[last_lon]))))
        #dist_nomask, idx_near_obs_nomask = nemo_tree_nomask.query(np.column_stack((float(obs[last_lat]),float(obs[last_lon]))))
        #ilat_obs, ilon_obs = np.unravel_index(idx_near_obs_nomask, U_current.nav_lat.shape)

        print("obs_point:", obs, depth_osservazione)
        print("nearest point:", nemo_lonlat[idx_near_obs], U_current.deptht.values[idx_near_obs_depth])
        #print("nearest point nomask:", U_current.nav_lat.values[ilat_obs, ilon_obs], U_current.nav_lon.values[ilat_obs, ilon_obs], U_current.deptht.values[idx_near_obs_depth])
        print("horizontal distance: ", dist)
        print("vertical distance: ", dist_depth)
        #print("horizontal distance nomask: ", dist_nomask)
        print("vertical distance nomask: ", dist_depth)

        i = np.where(vettore_lon==nemo_lonlat[idx_near_obs][0][1])
        j = np.where(vettore_lat==nemo_lonlat[idx_near_obs][0][0])
        k = np.where(vettore_depth==U_current.deptht.values[idx_near_obs_depth])
        nearest_point[name] = {'lon_idx': i, 'lat_idx': j, 'depth_idx': k}
        U_current.close()

    return nearest_point

def save_nc_mod_ts(obs_file, start_date, end_date, name_exp, date_in, date_fin, temp_res, path_to_destag_output_folder, nearest_point, path_to_output_mod_ts_folder):
    mod_file = {}
    for name, obs in obs_file.items():
        print(name)
        for single_date in daterange(start_date, end_date):
            print(single_date.strftime("%Y-%m-%d"))
            timetag = single_date.strftime("%Y%m%d")
            counter=0

            for exp_name in name_exp:
                destaggered_u_file= exp_name + "_" + temp_res + "_" + timetag + "_grid_U2T.nc"
                destaggered_v_file= exp_name + "_" + temp_res + "_" + timetag + "_grid_V2T.nc"
                try:
                    U_current = xr.open_dataset(path_to_destag_output_folder + destaggered_u_file)
                    V_current = xr.open_dataset(path_to_destag_output_folder + destaggered_v_file)
                except Exception:
                    counter=counter+1
                    continue
            if counter==len(name_exp):
                append_value(mod_file, name, np.nan)
                continue

            i = nearest_point[name]['lon_idx'][0][0]
            j = nearest_point[name]['lat_idx'][0][0]
            k = nearest_point[name]['depth_idx'][0][0]
            u = U_current.destaggered_u.values[k,j,i]
            U_current.close()
            v = V_current.destaggered_v.values[k,j,i]
            V_current.close()
            velocity = math.sqrt(u**2 + v**2)
            append_value(mod_file, name, velocity)

        array_name = np.array([obs[name_stat], obs[CMEMS_code], obs[WMO_code]])
        boolArr = np.where(array_name != "_")
        output_mod_file = array_name[boolArr][0] + "_" + date_in + "_" + date_fin + "_" + temp_res + "_mod.nc"
        f = nc4.Dataset(path_to_output_mod_ts_folder + output_mod_file,'w', format='NETCDF4')
        velgrp = f.createGroup('current velocity time series')
        velgrp.createDimension('time', None)
        vel = velgrp.createVariable('Current Velocity', 'f4', 'time')
        vel[:] = mod_file[name]
        vel.units = 'm/s'
        f.close()

if __name__ == "__main__":
    print('Argument entered:', str(sys.argv))

    argv=sys.argv
    temp_res=argv[1]
    date_in=argv[2]
    date_fin=argv[3]
    path_to_mod_output=argv[4]
    path_to_obs_file=argv[5]
    path_to_mask_file=argv[6]
    path_to_destag_output_folder=argv[7]
    path_to_output_mod_ts_folder=argv[8]
    depth_obs=argv[9]
    name_exp=argv[10:]

    print("name experiment: ", name_exp)
    print("temporal resolution: ", temp_res)
    print("date_in: ", date_in)
    print("date_fin: ", date_fin)
    print("path to model output files: ", path_to_mod_output)
    print("path to obs folder: ", path_to_obs_file)
    print("path to mask file: ", path_to_mask_file)
    print("path to destaggered model files folder: ", path_to_destag_output_folder)
    print("path to output model time series folder: ", path_to_output_mod_ts_folder)
    print("observation depth chosen: ", depth_obs)

    os.makedirs(path_to_output_mod_ts_folder, exist_ok=True)
    os.makedirs(path_to_destag_output_folder, exist_ok=True)

    directory = os.listdir(path_to_destag_output_folder)
    if len(directory) == 0:
        #produce and save destaggered files
        print("destaggering starts")
        destaggering(date_in, date_fin, path_to_mod_output, path_to_destag_output_folder, name_exp, temp_res, path_to_mask_file)
        print("destaggering ends")

    mesh_mask = xr.open_dataset(path_to_mask_file)
    t_mask = mesh_mask.tmask.values
    t_mask = np.squeeze(t_mask[0, :, :, :])

    start_date = date(int(date_in[0:4]),int(date_in[4:6]) , int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]),int(date_fin[4:6]) , int(date_fin[6:8]))

    obs_file = {}

    with open(path_to_obs_file) as f:
        first_line = f.readline()
        column_names = first_line.replace("#","")
        column_names = column_names.split(';')
        column_names.append('depth')
        print(column_names)
        reader = csv.reader(filter(lambda row: row[0]!='#',f), delimiter=';')
        for count, row in enumerate(reader):
            print(row)
            row.append(depth_obs)
            obs_file[count]= dict(zip(column_names, row))

    last_lat = column_names[0]
    last_lon = column_names[1]
    num_vlevs = column_names[2]
    num_sfc_levs = column_names[3]
    name_stat = column_names[4]
    CMEMS_code = column_names[5]
    WMO_code = column_names[6]
    time_period = column_names[7]
    fields_list = column_names[8]
    qf_value = column_names[9]
    path_to_obs_file = column_names[10]
    depth_obs = column_names[12]

    nearest_point = {}
    nearest_point = find_nearest_mod_point(obs_file, end_date, name_exp, temp_res, path_to_destag_output_folder, depth_obs, t_mask)
    save_nc_mod_ts(obs_file, start_date, end_date, name_exp, date_in, date_fin, temp_res, path_to_destag_output_folder, nearest_point, path_to_output_mod_ts_folder)
