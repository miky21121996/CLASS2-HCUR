import sys
import netCDF4 as nc4
import os.path
from os import listdir
from os.path import isfile, join
from datetime import datetime
import math
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from datetime import timedelta, date
import csv
from pandas import DatetimeIndex
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
mpl.use('Agg')

argv=sys.argv
date_in=argv[1]
date_fin=argv[2]
path_to_csv_file=argv[3]
time_res=argv[4]
path_to_out_obs_ts=argv[5]
depth_obs=argv[6]
nan_treshold=argv[7]
print("nan_treshold: ", nan_treshold)
path_to_accepted_obs_file=argv[8]


isExist = os.path.exists(path_to_out_obs_ts)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_to_out_obs_ts)
    print("The new directory is created!")

ini_date = date(int(date_in[0:4]),int(date_in[4:6]) , int(date_in[6:8]))
fin_date = date(int(date_fin[0:4]),int(date_fin[4:6]) , int(date_fin[6:8]))
delta = fin_date - ini_date
n_days = delta.days + 1
print("number of days is: ", n_days)

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def Check_Time_Period(start_date, end_date, obs_file):
    print(start_date)
    #start_date = start_date[:4] + '-' + start_date[4:]
    #start_date = start_date[:7] + '-' + start_date[7:]
    start_date = datetime.strptime(start_date, "%Y%m%d")
    #start_date = datetime.strptime(start_date, "%y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    for count in obs_file.copy():
        time_key = obs_file[count][time_period].split(',')
        csv_start_date = datetime.strptime(time_key[0], "%Y-%m-%d")
        csv_end_date = datetime.strptime(time_key[1], "%Y-%m-%d")
        if (csv_start_date <= start_date <= csv_end_date and csv_start_date <= end_date <= csv_end_date):
            pass
            print(obs_file[count][name] + ' has a time period that contains the chosen one')
        else:
            print(obs_file[count][name] + ' does not have a time period that contains the chosen one. I will remove it')
            obs_file.pop(count)
    return obs_file

def Check_N_Sfc_Levs(obs_file):
    for count in obs_file.copy():
        num_surface_levs = obs_file[count][num_sfc_levs]
        if num_surface_levs == '0':
            print(obs_file[count][name] + ' does not have a surface levels. I will remove it')
            obs_file.pop(count)
    return obs_file

def Check_Fields(obs_file):
    for count in obs_file.copy():
        fields = obs_file[count][fields_list]
        fields_splitted = fields.split(',')
        if not 'HCSP' in fields_splitted:
            if not 'EWCT' in fields_splitted or not 'NSCT' in fields_splitted:
                print(obs_file[count][name] + ' does not have necessary field. I will remove it')
                obs_file.pop(count)
    return obs_file

def Check_Depth(start, end, obs_file, dict_daily_vel_obs, dict_count_nan_vel_obs, q_flag, depth_array, qflag_array):
    depth_ts={}
    qflag_ts={}
    for count in obs_file.copy():
        array_name = np.array([obs_file[count][name], obs_file[count][CMEMS_code], obs_file[count][WMO_code]])
        boolArr_name = np.where(array_name != "_")
        print("name station: ", array_name[boolArr_name][0])
        dataset_obs = xr.open_dataset(obs_file[count][path_to_obs_file])
        time_array = list(dataset_obs["TIME"][:].values)
        days = np.array([np.datetime_as_string(i, unit='D').split('T', 1)[0] for i in time_array])
        print("istanti temporali: ", time_array[37601:40526])
        
        depth_ts[array_name[boolArr_name][0]]=depth_array
        qflag_ts[array_name[boolArr_name][0]]=qflag_array

        for day_index, single_date in enumerate(daterange(start, end)):
            print(single_date.strftime("%Y-%m-%d")) 
            timetag = single_date.strftime("%Y-%m-%d")
            boolArr = np.argwhere(days == timetag)
            print("boolArr: ",boolArr)
            print("days di boolArr: ",days[boolArr])
            if boolArr.shape[0] == 0:
                print("not contained in observation: ", single_date)
#                depth_ts[count] = np.append(depth_ts[count],np.NaN)
                depth_ts[array_name[boolArr_name][0]].append(np.NaN)
#                qflag_ts[count] = np.append(qflag_ts[count],np.NaN)
                qflag_ts[array_name[boolArr_name][0]].append(np.NaN)
                continue              
            else:
                print("boolArr shape: ", boolArr.shape[0])
                print("boolArr first element: ",boolArr[0][0]) 
                print("boolArr last element: ", boolArr[-1][0]) 
                vel_obs_time_instants = np.empty([boolArr.shape[0],1])
                vel_obs_time_instants[:] = np.NaN
                for instant_counter, i in enumerate(range(boolArr[0][0],boolArr[-1][0] + 1)):                
                    print("instant_counter: ",instant_counter)
                    print("indice in questione: ",i)
                    index_depth = np.argwhere(np.array(dataset_obs["DEPH"][i,:].values)==3)
                    if index_depth.shape[0] != 0:    
                        print("depth " + depth_obs + " exists in: ", dataset_obs["TIME"][boolArr[instant_counter]].values)
                        fields = obs_file[count][fields_list]
                        fields_splitted = fields.split(',')
                        print(fields_splitted)
                        if 'HCSP' in fields_splitted:
                            print("quality flag HCSP: ", dataset_obs["HCSP_QC"][i,index_depth[0]].values)
#                            depth_ts[count] = np.append(depth_ts[count],dataset_obs["DEPH"][i,index_depth[0]].values)
                            depth_ts[array_name[boolArr_name][0]].append(dataset_obs["DEPH"][i,index_depth[0]].values)
                            #qflag_ts[count] = np.append(qflag_ts[count],dataset_obs["HCSP_QC"][i,index_depth[0]].values)
                            qflag_ts[array_name[boolArr_name][0]].append(dataset_obs["HCSP_QC"][i,index_depth[0]].values)
                            if dataset_obs["HCSP_QC"][i,index_depth[0]].values == int(float(obs_file[count][q_flag])):                                
                                vel_obs_time_instants[instant_counter] = dataset_obs["HCSP"][i,index_depth[0]].values
                                print("sono dentro check HCSP: ", instant_counter)
                                print("sono dentro check HCSP: ", i)
                                continue
                            else:
                                print("quality check for HCSP not passed for {} at {}".format(array_name[boolArr_name][0],dataset_obs["TIME"][i].values))
                        print('EWCT' in fields_splitted)
                        if 'EWCT' in fields_splitted:
                            print("quality flag EWCT: ", dataset_obs["EWCT_QC"][i,index_depth[0]].values)
                            print("quality flag NSCT: ", dataset_obs["NSCT_QC"][i,index_depth[0]].values)
                            
                            if dataset_obs["EWCT_QC"][i,index_depth[0]].values == int(float(obs_file[count][q_flag])) and dataset_obs["NSCT_QC"][i,index_depth[0]].values == int(float(obs_file[count][q_flag])):
                                #depth_ts[count] = np.append(depth_ts[count],dataset_obs["DEPH"][i,index_depth[0]].values)
                                depth_ts[array_name[boolArr_name][0]].append(dataset_obs["DEPH"][i,index_depth[0]].values)
                                #qflag_ts[count] = np.append(qflag_ts[count],dataset_obs["EWCT_QC"][i,index_depth[0]].values)
                                qflag_ts[array_name[boolArr_name][0]].append(dataset_obs["EWCT_QC"][i,index_depth[0]].values)
                                u = dataset_obs["EWCT"][i,index_depth[0]].values
                                v = dataset_obs["NSCT"][i,index_depth[0]].values
                                vel_obs_time_instants[instant_counter] = math.sqrt(u**2 + v**2)
                                print("sono dentro check EWCT: ", instant_counter)
                                print("sono dentro check EWCT: ", i)
                                continue
                            else:
                                #depth_ts[count] = np.append(depth_ts[count],np.NaN)
                                depth_ts[array_name[boolArr_name][0]].append(np.NaN)
                                #qflag_ts[count] = np.append(qflag_ts[count],np.NaN)
                                qflag_ts[array_name[boolArr_name][0]].append(np.NaN)
                                print("quality check for EWCT and/or NSCT not passed for {} at {}".format(array_name[boolArr_name][0],dataset_obs["TIME"][i].values))
                                continue
                    else:
                        print("no depth " + depth_obs + ", looking for close depths")
                        index_depth = np.argwhere(np.logical_and(np.array(dataset_obs["DEPH"][i,:].values) >= (float(depth_obs) - 1), np.array(dataset_obs["DEPH"][i,:].values) <= (float(depth_obs) + 1)))
                        if index_depth.shape[0] != 0:
                            print("index depth: ",index_depth)
                            dist=[]
                            for depth_count in index_depth:
                                dist.append(abs(dataset_obs["DEPH"][i,depth_count].values-float(depth_obs)))
                                print("depth value: ", dataset_obs["DEPH"][i,depth_count].values)
                            dist=np.array(dist)
                            print("dist: ", dist)
                            closest_index_depth = np.argwhere(dist==np.min(dist))[0][0]
                            print("close index depth: ",index_depth[closest_index_depth])
                            print("found close depth at: ", dataset_obs["DEPH"][i,index_depth[closest_index_depth]].values)
                            fields = obs_file[count][fields_list]
                            fields_splitted = fields.split(',')
                            if 'HCSP' in fields_splitted:
                                print("quality flag HCSP: ", dataset_obs["HCSP_QC"][i,index_depth[closest_index_depth]].values)
                                #depth_ts[count] = np.append(depth_ts[count],dataset_obs["DEPH"][i,index_depth[closest_index_depth]].values)
                                depth_ts[array_name[boolArr_name][0]].append(dataset_obs["DEPH"][i,index_depth[closest_index_depth]].values)
                                #qflag_ts[count] = np.append(qflag_ts[count],dataset_obs["HCSP_QC"][i,index_depth[closest_index_depth]].values)
                                qflag_ts[array_name[boolArr_name][0]].append(dataset_obs["HCSP_QC"][i,index_depth[closest_index_depth]].values)
                                if dataset_obs["HCSP_QC"][i,index_depth[closest_index_depth]].values == int(float(obs_file[count][q_flag])):
                                    vel_obs_time_instants[instant_counter] = dataset_obs["HCSP"][i,index_depth[closest_index_depth]].values
                                    print("sono dentro check HCSP depth vicine: ", instant_counter)
                                    print("sono dentro check HCSP depth vicine: ", i)
                                    continue
                                else:
                                    print("quality check for HCSP not passed for {} at {}".format(array_name[boolArr_name][0], dataset_obs["TIME"][i].values))
                            if 'EWCT' in fields_splitted:
                                print("quality flag EWCT: ", dataset_obs["EWCT_QC"][i,index_depth[closest_index_depth]].values)
                                print("quality flag NSCT: ", dataset_obs["NSCT_QC"][i,index_depth[closest_index_depth]].values)
                                if dataset_obs["EWCT_QC"][i,index_depth[closest_index_depth]].values == int(float(obs_file[count][q_flag])) and dataset_obs["NSCT_QC"][i,index_depth[closest_index_depth]].values == int(float(obs_file[count][q_flag])):
                                    #depth_ts[count] = np.append(depth_ts[count],dataset_obs["DEPH"][i,index_depth[closest_index_depth]].values)
                                    depth_ts[array_name[boolArr_name][0]].append(dataset_obs["DEPH"][i,index_depth[closest_index_depth]].values)
                                    #qflag_ts[count] = np.append(qflag_ts[count],dataset_obs["EWCT_QC"][i,index_depth[closest_index_depth]].values)
                                    qflag_ts[array_name[boolArr_name][0]].append(dataset_obs["EWCT_QC"][i,index_depth[closest_index_depth]].values)
                                    u = dataset_obs["EWCT"][i,index_depth[closest_index_depth]].values
                                    v = dataset_obs["NSCT"][i,index_depth[closest_index_depth]].values
                                    vel_obs_time_instants[instant_counter] = math.sqrt(u**2 + v**2)
                                    print("sono dentro check EWCT depth vicine: ", instant_counter)
                                    print("sono dentro check EWCT depth vicine: ", i)
                                    continue
                                else:
                                    print("quality check for EWCT and/or NSCT not passed for {} at {}".format(array_name[boolArr_name][0], dataset_obs["TIME"][i].values))
                                    #depth_ts[count] = np.append(depth_ts[count],np.NaN)
                                    depth_ts[array_name[boolArr_name][0]].append(np.NaN)
                                    #qflag_ts[count] = np.append(qflag_ts[count],np.NaN)
                                    qflag_ts[array_name[boolArr_name][0]].append(np.NaN)
                                    continue
                        else:
                            print("no surface depth found")
                            #depth_ts[count] = np.append(depth_ts[count],np.NaN)
                            depth_ts[array_name[boolArr_name][0]].append(np.NaN)
                            #qflag_ts[count] = np.append(qflag_ts[count],np.NaN)
                            qflag_ts[array_name[boolArr_name][0]].append(np.NaN)
                            continue                   
                #if not vel_obs_time_instants:
                dict_daily_vel_obs[count][day_index] = np.nanmean(vel_obs_time_instants[:])
                dict_count_nan_vel_obs[count][day_index] = np.count_nonzero(np.isnan(vel_obs_time_instants[:]))/len(vel_obs_time_instants[:])       
        print(dict_daily_vel_obs[count])
        print(dict_count_nan_vel_obs[count])
        print(depth_ts[array_name[boolArr_name][0]])
        print(qflag_ts[array_name[boolArr_name][0]])

    return dict_daily_vel_obs, dict_count_nan_vel_obs, depth_ts, qflag_ts

def check_nan(dict_daily_vel_obs, dict_count_nan_vel_obs, obs_file, n_days):
    for count in obs_file.copy():
#        for day_index in range(n_days):
#            if dict_count_nan_vel_obs[count][day_index] > 0.5:
#                dict_daily_vel_obs[count][day_index] = np.NaN
        nan_counter = np.count_nonzero(np.isnan(dict_daily_vel_obs[count]))/len(dict_daily_vel_obs[count])
        print("nan counter: ", nan_counter)
        if nan_counter > float(nan_treshold):
             obs_file.pop(count)
             dict_daily_vel_obs.pop(count)
    return obs_file, dict_daily_vel_obs

obs_file = {}

with open(path_to_csv_file) as f:
    first_line = f.readline()
    column_names = first_line.replace("#","")
    column_names = column_names.split(';')
    reader = csv.reader(filter(lambda row: row[0]!='#',f), delimiter=';')
    for count, row in enumerate(reader):
        obs_file[count]= dict(zip(column_names, row))

print(len(obs_file.keys()))

last_lat = column_names[0]
last_lon = column_names[1]
num_vlevs = column_names[2]
num_sfc_levs = column_names[3]
name = column_names[4]
CMEMS_code = column_names[5]
WMO_code = column_names[6]
time_period = column_names[7]
fields_list = column_names[8]
qf_value = column_names[9]
path_to_obs_file = column_names[10]


obs_file = Check_Time_Period(date_in,date_fin,obs_file)
print("number of stations after check of time period: ", len(obs_file.keys()))

obs_file = Check_N_Sfc_Levs(obs_file)
print("number of stations after check of number of surface levels: ", len(obs_file.keys()))

obs_file = Check_Fields(obs_file)
print("number of stations after check of fields: ", len(obs_file.keys()))

 
dict_daily_vel_obs = {}
dict_count_nan_vel_obs = {}
for count in obs_file.keys():
    a = np.empty([n_days,1])
    a[:] = np.NaN
    b = np.empty([n_days,1])
    b[:] = 1
    dict_daily_vel_obs[count] = a
    dict_count_nan_vel_obs[count] = b

level=[]
flag=[]
#dict_depth={}
#dict_qflag={}
dict_daily_vel_obs, dict_count_nan_vel_obs, depth_ts, qflag_ts = Check_Depth(ini_date, fin_date, obs_file, dict_daily_vel_obs, dict_count_nan_vel_obs, qf_value, level, flag)

obs_file, dict_daily_vel_obs = check_nan(dict_daily_vel_obs, dict_count_nan_vel_obs, obs_file, n_days)

print(dict_daily_vel_obs)

for count, obs in obs_file.items():
    array_name = np.array([obs[name], obs[CMEMS_code], obs[WMO_code]])
    boolArr = np.where(array_name != "_")
    output_obs_file = array_name[boolArr][0] + "_" + date_in + "_" + date_fin + "_" + time_res + "_obs.nc"
    f = nc4.Dataset(path_to_out_obs_ts + output_obs_file,'w', format='NETCDF4')
    velgrp = f.createGroup('current velocity time series')
    velgrp.createDimension('time', None)
    vel = velgrp.createVariable('Current Velocity', 'f4', 'time')
    vel[:] = dict_daily_vel_obs[count]
    vel.units = 'm/s'
    depthgrp = f.createGroup('depth time series')
    depthgrp.createDimension('time', None)
    depth = depthgrp.createVariable('Depth', 'f4', 'time')
    depth[:] = depth_ts[array_name[boolArr][0]]
    depth.units = 'm'
    qflaggrp = f.createGroup('qflag time series')
    qflaggrp.createDimension('time', None)
    qflag = qflaggrp.createVariable('Qflag', 'f4', 'time')
    qflag[:] = qflag_ts[array_name[boolArr][0]]
    f.close()

with open(path_to_accepted_obs_file, "w") as f:
    f.write(first_line)
    w = csv.writer(f, delimiter = ";")
    obs_infos = list(list(obs_file.values())[0].keys())
    for key in obs_file.keys():
        w.writerow([obs_file[key][obs_info] for obs_info in obs_infos])


text = open(path_to_accepted_obs_file, "r") 
# csvfile.csv and formed as a string
text = ''.join([i for i in text]) 
# search and replace the contents
text = text.replace('"', "") 
x = open(path_to_accepted_obs_file,"w")
x.writelines(text)
x.close() 
