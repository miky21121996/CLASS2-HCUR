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
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
mpl.use('Agg')
from destaggering_UV import destaggering

print('Argument entered:', str(sys.argv))
if (len(sys.argv)-1 != 10):
    print('number of arguments:', len(sys.argv)-1, 'arguments, needed 10 arguments')
    sys.exit()

argv=sys.argv
name_exp=argv[1]
temp_res=argv[2]
date_in=argv[3]
date_fin=argv[4]
path_to_mod_output=argv[5]
path_to_obs_file=argv[6]
path_to_mask_file=argv[7]
path_to_destag_output_folder=argv[8]
path_to_output_mod_ts_folder=argv[9]
depth_obs=argv[10]

isExist = os.path.exists(path_to_output_mod_ts_folder)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_to_output_mod_ts_folder)
    print("The new directory is created!")

isExist = os.path.exists(path_to_destag_output_folder)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_to_destag_output_folder)
    print("The new directory is created!")

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
#filename = "MO_curr_ok.csv"
#with open(path_to_obs_folder + filename) as f:
#    reader = csv.reader(f,delimiter=';')
#    for row in reader:
#        print(row)
#        obs_file[row[3]] = {'lat': row[0], 'lon': row[1], 'depth': row[2]}

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

for k, v in obs_file.items():
    print(k, v)

nearest_point = {}    
for name, obs in obs_file.items():
#    if (name!='LA-MOLA'):
#        continue
    print(name)
    print(start_date.strftime("%Y-%m-%d"))
    timetag = end_date.strftime("%Y%m%d")

    destaggered_u_file= name_exp + "_" + temp_res + "_" + timetag + "_grid_U2T.nc"
    destaggered_v_file= name_exp + "_" + temp_res + "_" + timetag + "_grid_V2T.nc"

    U_current = xr.open_dataset(path_to_destag_output_folder + destaggered_u_file)
    V_current = xr.open_dataset(path_to_destag_output_folder + destaggered_v_file)

    vettore_lat=U_current.nav_lat.data[:,0]
    vettore_lon=U_current.nav_lon.data[0,:]
    vettore_depth=U_current.depthu.data

    nemo_lonlat_nomask = np.column_stack((U_current.nav_lat.values.ravel(), U_current.nav_lon.values.ravel()))
    nemo_tree_nomask = cKDTree(nemo_lonlat_nomask)
    print(obs)    
    print("ciao: ", obs[depth_obs])
    depth_osservazione = float(obs[depth_obs])
    array_depth = np.column_stack(U_current.depthu.values)
    nemo_tree = cKDTree(array_depth.T)
    dist_depth, idx_near_obs_depth = nemo_tree.query(np.array([depth_osservazione]))

    masked_nav_lat = U_current.nav_lat.values * np.squeeze(t_mask[idx_near_obs_depth,:,:])
    masked_nav_lon = U_current.nav_lon.values * np.squeeze(t_mask[idx_near_obs_depth,:,:])
    masked_nav_lat[masked_nav_lat==0] = np.nan
    masked_nav_lon[masked_nav_lon==0] = np.nan

    nemo_lonlat = np.column_stack((masked_nav_lat.ravel(), masked_nav_lon.ravel()))
    nemo_lonlat = nemo_lonlat[~np.isnan(nemo_lonlat).any(axis=1)]

    nemo_tree = cKDTree(nemo_lonlat)

    print(name)
    dist, idx_near_obs = nemo_tree.query(np.column_stack((float(obs[last_lat]), float(obs[last_lon]))))
    dist_nomask, idx_near_obs_nomask = nemo_tree_nomask.query(np.column_stack((float(obs[last_lat]),float(obs[last_lon]))))
    ilat_obs, ilon_obs = np.unravel_index(idx_near_obs_nomask, U_current.nav_lat.shape)
    print("obs_point:", obs, depth_osservazione)
    print("nearest point:", nemo_lonlat[idx_near_obs], U_current.depthu.values[idx_near_obs_depth])
    print("nearest point nomask:", U_current.nav_lat.values[ilat_obs, ilon_obs], U_current.nav_lon.values[ilat_obs, ilon_obs], U_current.depthu.values[idx_near_obs_depth])
    print("horizontal distance: ", dist)
    print("vertical distance: ", dist_depth)
    print("horizontal distance nomask: ", dist_nomask)
    print("vertical distance nomask: ", dist_depth)
    i = np.where(vettore_lon==nemo_lonlat[idx_near_obs][0][1])
    j = np.where(vettore_lat==nemo_lonlat[idx_near_obs][0][0])
    k = np.where(vettore_depth==U_current.depthu.values[idx_near_obs_depth])
    nearest_point[name] = {'lon_idx': i, 'lat_idx': j, 'depth_idx': k}

mod_file = {}
for name, obs in obs_file.items():
    print(name)
    for single_date in daterange(start_date, end_date):
        print(single_date.strftime("%Y-%m-%d"))
        timetag = single_date.strftime("%Y%m%d")

        destaggered_u_file= name_exp + "_" + temp_res + "_" + timetag + "_grid_U2T.nc"
        destaggered_v_file= name_exp + "_" + temp_res + "_" + timetag + "_grid_V2T.nc"
        try:
            U_current = xr.open_dataset(path_to_destag_output_folder + destaggered_u_file)
            V_current = xr.open_dataset(path_to_destag_output_folder + destaggered_v_file)
        except Exception:
            append_value(mod_file, name, np.nan)
            continue
        #U_current = xr.open_dataset(path_to_destag_output_folder + destaggered_u_file)
        #V_current = xr.open_dataset(path_to_destag_output_folder + destaggered_v_file)
        i = nearest_point[name]['lon_idx'][0][0]
        j = nearest_point[name]['lat_idx'][0][0]
        k = nearest_point[name]['depth_idx'][0][0]
        print(i,j,k)
        print("model u at nearest point: ", U_current.destaggered_u.values[k,j,i])
        print("model v at nearest point: ", V_current.destaggered_v.values[k,j,i])
#        print("model u at nearest point no mask: ", U_current.destaggered_u.values[k,ilat_obs,ilon_obs])
#        print("model v at nearest point no mask: ", V_current.destaggered_v.values[k,ilat_obs,ilon_obs])
        u = U_current.destaggered_u.values[k,j,i]
        v = V_current.destaggered_v.values[k,j,i]
        velocity = math.sqrt(u**2 + v**2)
        print("model velocity at nearest point: ", velocity)
        append_value(mod_file, name, velocity)
#   mod_file[name].append(vel)
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


#plt.figure(figsize=(12,8))
#map = Basemap(projection='cyl',llcrnrlat=30.19,urcrnrlat=45.98,llcrnrlon=-18.125,urcrnrlon=36.3,resolution='h')
#g = np.linspace(-0.0000001,0.0000001, 41, endpoint=True)
#cs = map.contourf(lon,lat,U_current.nav_lon.values, cmap=plt.cm.jet,extend='both')
#map.drawcoastlines()
#map.drawstates()
#map.drawcountries()
#map.fillcontinents(color='gray')
#parallels = np.arange(10, 50 ,5.) # make latitude lines ever 5 degrees from 30N-50N
#meridians = np.arange(-20, 50 ,5.) # make longitude lines every 5 degrees from 95W to 70W
#map.drawparallels(parallels,labels=[1,0,0,0],fontsize=14)
#map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14)
#plt.title('nav lon meshgrid',fontsize=20)
#cbar = map.colorbar(cs,location='right',size='2%',pad="3%")
#cbar.ax.tick_params(labelsize=18)
#cbar.set_label('Depth [m]',fontweight='bold')
#plt.savefig('nav_lon_meshgrid.png')

#plt.figure(figsize=(12,8))
#map = Basemap(projection='cyl',llcrnrlat=30.19,urcrnrlat=45.98,llcrnrlon=-18.125,urcrnrlon=36.3,resolution='h')
#g = np.linspace(-0.0000001,0.0000001, 41, endpoint=True)
#cs = map.contourf(lon,lat,U_current.nav_lat.values, cmap=plt.cm.jet,extend='both')
#map.drawcoastlines()
#map.drawstates()
#map.drawcountries()
#map.fillcontinents(color='gray')
#parallels = np.arange(10, 50 ,5.) # make latitude lines ever 5 degrees from 30N-50N
#meridians = np.arange(-20, 50 ,5.) # make longitude lines every 5 degrees from 95W to 70W
#map.drawparallels(parallels,labels=[1,0,0,0],fontsize=14)
#map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14)
#plt.title('nav lat meshgrid',fontsize=20)
#cbar = map.colorbar(cs,location='right',size='2%',pad="3%")
#cbar.ax.tick_params(labelsize=18)
#cbar.set_label('Depth [m]',fontweight='bold')
#plt.savefig('nav_lat_meshgrid.png')


#plt.figure(figsize=(12,8))
#map = Basemap(projection='cyl',llcrnrlat=30.19,urcrnrlat=45.98,llcrnrlon=-18.125,urcrnrlon=36.3,resolution='h')
#g = np.linspace(-0.0000001,0.0000001, 41, endpoint=True)
#cs = map.contourf(lon,lat,np.squeeze(t_mask[0,:,:]), cmap=plt.cm.jet,extend='both')
#map.drawcoastlines()
#map.drawstates()
#map.drawcountries()
#map.fillcontinents(color='gray')
#parallels = np.arange(10, 50 ,5.) # make latitude lines ever 5 degrees from 30N-50N
#meridians = np.arange(-20, 50 ,5.) # make longitude lines every 5 degrees from 95W to 70W
#map.drawparallels(parallels,labels=[1,0,0,0],fontsize=14)
#map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14)
#plt.title('t mask',fontsize=20)
#cbar = map.colorbar(cs,location='right',size='2%',pad="3%")
#cbar.ax.tick_params(labelsize=18)
#cbar.set_label('Depth [m]',fontweight='bold')
#plt.savefig('t_mask.png')


#plt.figure(figsize=(12,8))
#map = Basemap(projection='cyl',llcrnrlat=30.19,urcrnrlat=45.98,llcrnrlon=-18.125,urcrnrlon=36.3,resolution='h')
#g = np.linspace(-0.0000001,0.0000001, 41, endpoint=True)
#cs = map.contourf(lon,lat,masked_nav_lat, cmap=plt.cm.jet,extend='both')
#map.drawcoastlines()
#map.drawstates()
#map.drawcountries()
#map.fillcontinents(color='gray')
#parallels = np.arange(10, 50 ,5.) # make latitude lines ever 5 degrees from 30N-50N
#meridians = np.arange(-20, 50 ,5.) # make longitude lines every 5 degrees from 95W to 70W
#map.drawparallels(parallels,labels=[1,0,0,0],fontsize=14)
#map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14)
#plt.title('Masked nav lat',fontsize=20)
#cbar = map.colorbar(cs,location='right',size='2%',pad="3%")
#cbar.ax.tick_params(labelsize=18)
#cbar.set_label('Depth [m]',fontweight='bold')
#plt.savefig('masked_nav_lat.png')

#plt.figure(figsize=(12,8))
#map = Basemap(projection='cyl',llcrnrlat=30.19,urcrnrlat=45.98,llcrnrlon=-18.125,urcrnrlon=36.3,resolution='h')
#g = np.linspace(-0.0000001,0.0000001, 41, endpoint=True)
#cs = map.contourf(lon,lat,masked_nav_lon, cmap=plt.cm.jet,extend='both')
#map.drawcoastlines()
#map.drawstates()
#map.drawcountries()
#map.fillcontinents(color='gray')
#parallels = np.arange(10, 50 ,5.) # make latitude lines ever 5 degrees from 30N-50N
#meridians = np.arange(-20, 50 ,5.) # make longitude lines every 5 degrees from 95W to 70W
#map.drawparallels(parallels,labels=[1,0,0,0],fontsize=14)
#map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14)
#plt.title('Masked nav lon',fontsize=20)
#cbar = map.colorbar(cs,location='right',size='2%',pad="3%")
#cbar.ax.tick_params(labelsize=18)
#cbar.set_label('Depth [m]',fontweight='bold')
#plt.savefig('masked_nav_lon.png')

#for name, obs in obs_file.items():
#    print(name)
#    dist, idx_near_obs = nemo_tree.query(np.column_stack((float(obs[0]),float(obs[1]))))
#    ilat_obs, ilon_obs = np.unravel_index(idx_near_obs, U_current.nav_lat.shape)
#    print("obs_point: ", obs)
#    print("nearest lat: ", U_current.nav_lat.values[ilat_obs, ilon_obs])
#    print("nearest lon: ", U_current.nav_lon.values[ilat_obs, ilon_obs])
#    print("distance: ", dist)
#    print("masked lat: ", masked_nav_lat[ilat_obs, ilon_obs])
#    print("masked lon: ", masked_nav_lon[ilat_obs, ilon_obs])

#nemo_lonlat_a = np.column_stack((masked_nav_lat.ravel(), masked_nav_lon.ravel()))
#nemo_tree_a = cKDTree(nemo_lonlat_a)
#for name, obs in obs_file.items():
#    print(name)
#    dist, idx_near_obs = nemo_tree_a.query(np.column_stack((float(obs[0]),float(obs[1]))))
#    ilat_obs, ilon_obs = np.unravel_index(idx_near_obs, U_current.nav_lat.shape)   
#    print("obs_point: ", obs)
#    print("nearest lat: ", U_current.nav_lat.values[ilat_obs, ilon_obs])
#    print("nearest lon: ", U_current.nav_lon.values[ilat_obs, ilon_obs])
#    print("distance: ", dist)

#   print("coordinates from nav lat e lon: ", U_current.nav_lat.values[j,i], U_current.nav_lon.values[j,i])
