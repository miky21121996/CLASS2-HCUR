import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import netCDF4 as NC
import math
import os
from os import listdir
from os.path import isfile, join
import sys
import warnings
import numpy.ma as ma
warnings.filterwarnings("ignore") # Avoid warnings
from scipy.stats import linregress, pearsonr, gaussian_kde
from scipy.optimize import curve_fit
from scipy import stats
import csv
import math
import statistics
import pandas as pd
import datetime
from datetime import date, timedelta
import matplotlib.dates as mdates
from datetime import datetime
from astropy.visualization import hist
from matplotlib.colors import LogNorm
from statsmodels.distributions.empirical_distribution import ECDF # empirical distribution functions

def line_A(x, m_A, q_A):
    return (m_A*x+q_A)

def BIAS(data,obs):
    return  np.round((np.nanmean( data-obs)).data, 2)

def RMSE(data,obs):
    return np.round(np.sqrt(np.nanmean((data-obs)**2)),2)

def ScatterIndex(data,obs):
    num=np.sum(((data-np.nanmean(data))-(obs-np.nanmean(obs)))**2)
    denom=np.sum(obs**2)
    return np.round(np.sqrt((num/denom)),2)

def Normalized_std(data,obs):
    data_std=np.std(data)
    data_obs=np.std(obs)
    return np.round(data_std/data_obs,2)

def scatterPlot(mod, obs, outname, name, **kwargs):

    if np.isnan(obs).any() or np.isnan(mod).any():
        
        obs_no_nan = obs[~np.isnan(obs) & ~np.isnan(mod)]
        mod_no_nan = mod[~np.isnan(obs) & ~np.isnan(mod)]
        xy = np.vstack([obs_no_nan, mod_no_nan])
    else:
        xy = np.vstack([obs, mod])

    z = gaussian_kde(xy)(xy)
    idx = z.argsort()

    if np.isnan(obs).any() or np.isnan(mod).any():
        x, y, z = obs_no_nan[idx], mod_no_nan[idx], z[idx]
    else:
        x, y, z = obs[idx], mod[idx], z[idx]

    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.scatter(x, y, c=z, s=8, edgecolor=None, cmap='jet', clip_on=False)

    maxVal = np.nanmax((x, y))
    ax.set_ylim(0, maxVal)
    ax.set_xlim(0, maxVal)
    ax.set_aspect(1.0)
    ax.tick_params(axis='both', labelsize=12.5)

    bias = BIAS(y,x)
    corr, _ = pearsonr(x, y)
    rmse=RMSE(y,x)
    nstd=Normalized_std(y,x)
    si=ScatterIndex(y,x)
    slope,intercept, rvalue,pvalue,stderr=linregress(y,x)

    prova = x[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(prova, y)
    xseq = np.linspace(0, maxVal, num=100)
    ax.plot(xseq, a*xseq, 'r-')

    plt.text(0.12, 0.7, name[0], weight='bold',transform=plt.gcf().transFigure,fontsize=18)

    plt.text(0.12, 0.32, 'Entries: %s\n'
             'BIAS: %s m/s\n'
             'RMSD: %s m/s\n'
             'NSTD: %s\n'
             'SI: %s\n'
             'corr:%s\n'
             'Slope: %s\n'
             'STDerr: %s m/s'
             %(len(obs),bias,rmse,nstd,si,np.round(corr,2),
               np.round(a[0],2),np.round(stderr,2)),transform=plt.gcf().transFigure,fontsize=15)

    stat_array=[bias,rmse,si,np.round(corr,2),np.round(stderr,2),len(obs)]
    
    if 'title' in kwargs:
        plt.title(kwargs['title'], fontsize=20, x=0.5, y=1.01)

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'], fontsize=18)

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'], fontsize=18)


    ax.plot([0,maxVal],[0,maxVal],c='k',linestyle='-.')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ticks_1 = np.linspace(z.min(), z.max(), 5,endpoint=True)
    cbar=plt.colorbar(im,fraction=0.02,ticks=ticks_1)
    cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in ticks_1], fontsize=13)
    cbar.set_label('probaility density [%]', rotation=270,size=18,labelpad=15)

    plt.savefig(outname)
    plt.close()
    return stat_array

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def mapping(obs_file,lat,lon,work_dir_plot):

    for key in obs_file.copy():
        lat[key]=obs_file[key]['lat']
        lon[key]=obs_file[key]['lon']

    plt.figure(figsize=(12,8))
    m = Basemap(projection='cyl',llcrnrlat=30.19,urcrnrlat=45.98,llcrnrlon=-18.125,urcrnrlon=36.3,resolution='h')
    g = np.linspace(-0.0000001,0.0000001, 41, endpoint=True)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    m.fillcontinents(color='gray')
    parallels = np.arange(10, 50 ,5.) # make latitude lines ever 5 degrees from 30N-50N
    meridians = np.arange(-20, 50 ,5.) # make longitude lines every 5 degrees from 95W to 70W
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=14)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14)

    lons=[float(x) for x in list(lon.values())]
    lats=[float(x) for x in list(lat.values())]
    x, y= m(lons,lats)

    colors = plt.cm.jet(np.linspace(0,1,len(x)))

    for count,key in enumerate(list(obs_file.keys())):
        plt.scatter(x[count], y[count], 50, alpha=1,c=colors[count],label=key,zorder=2)

    plt.legend(bbox_to_anchor=(0.5,-0.87), loc='lower center', prop={'size': 13})
    plt.title('Location of accepted mooring',fontsize=25)
    plt.savefig(work_dir_plot+'location.png')

def plot_mod_obs_ts_diff(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, time_res_xaxis, path_to_output_plot_folder):
                plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ts_difference.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=24)
                plt.title('Surface (3m) Current Veloity BIAS: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: '+ date_in + '-' + date_fin, fontsize=29)
                mean_vel_bias = round(np.nanmean(np.array(vel_mod_ts[name_stat])-np.array(vel_obs_ts[name_stat])),2)
                mean_vel_rmsd = round(math.sqrt(np.nanmean((np.array(vel_mod_ts[name_stat])-np.array(vel_obs_ts[name_stat]))**2)),2)

                plt.plot(timerange,vel_mod_ts[name_stat]-vel_obs_ts[name_stat],label = 'BIAS: '+str(mean_vel_bias)+' m/s', linewidth=3)
                plt.plot([], [], ' ', label = 'RMSD: '+str(mean_vel_rmsd)+' m/s')
                plt.grid()
                plt.text(0.17, 0.89, name_exp[0], weight='bold',transform=plt.gcf().transFigure,fontsize=22)
                ax.tick_params(axis='both', labelsize=26)
                if time_res_xaxis[1]=='w':
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
                if time_res_xaxis[1]=='m':
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
                if time_res_xaxis[1]=='y':
                    ax.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_xaxis[0])))

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
                fig.autofmt_xdate()
                plt.ylabel('Velocity Difference [m/s]', fontsize=40)
                plt.xlabel('Date', fontsize=40)
                plt.legend(prop={'size': 20}, framealpha=0.2)
                plt.savefig(path_to_output_plot_folder + plotname)
                plt.clf()

def plot_mod_obs_ts(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, time_res_xaxis, path_to_output_plot_folder):
                plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ts.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=24)
                plt.title('Surface (3m) Current Velocity: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: '+ date_in + '-' + date_fin, fontsize=29)
                mean_vel_mod = round(np.nanmean(np.array(vel_mod_ts[name_stat])),2)
                mean_vel_obs = round(np.nanmean(np.array(vel_obs_ts[name_stat])),2)
                tot_mean_stat=[mean_vel_mod,mean_vel_obs]
                plt.plot(timerange,vel_mod_ts[name_stat],label = 'Model (mean: '+str(mean_vel_mod)+' m/s)', linewidth=3, color='darkblue')
                plt.plot(timerange,vel_obs_ts[name_stat],label = 'Observation (mean: '+str(mean_vel_obs)+' m/s)', linewidth=3, color='orange')
                plt.grid()
                plt.text(0.17, 0.89, name_exp[0], weight='bold',transform=plt.gcf().transFigure,fontsize=22)
                ax.tick_params(axis='both', labelsize=26)
                if time_res_xaxis[1]=='w':
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
                if time_res_xaxis[1]=='m':
               #ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                if time_res_xaxis[1]=='y':
                    ax.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_xaxis[0])))

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
                fig.autofmt_xdate()
                plt.ylabel('Velocity [m/s]', fontsize=40)
                plt.xlabel('Date', fontsize=40)
                plt.legend(prop={'size': 30}, framealpha=0.2)
           # Save and close
                plt.savefig(path_to_output_plot_folder + plotname)
                return tot_mean_stat

def plot_depth_obs_hist(name_stat, date_in, date_fin, obs_file, key_obs_file, depth_obs_ts, path_to_output_plot_folder):
                plotname = name_stat + '_' + date_in + '_' + date_fin + '_depth_obs_histogram.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=16)
                plt.title('Observation Depth Frequency Distribution: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n --- Time Period: '+ date_in + '-' + date_fin, fontsize=26)
                #min_mod = np.amin(np.array(vel_mod_ts[name_stat]))
                #max_mod = np.amax(np.array(vel_mod_ts[name_stat]))
                #min_obs = np.amin(np.array(vel_obs_ts[name_stat]))
                #max_obs = np.amax(np.array(vel_obs_ts[name_stat]))
                bins = np.linspace(np.amin(depth_obs_ts[name_stat][~np.isnan(depth_obs_ts[name_stat])]), np.amax(depth_obs_ts[name_stat][~np.isnan(depth_obs_ts[name_stat])]),100)

                plt.hist(depth_obs_ts[name_stat][~np.isnan(depth_obs_ts[name_stat])], bins, label='depth_obs_hist')
                ax.tick_params(axis='both', labelsize=13)
                plt.xlabel ('depth [m]', fontsize=40)
                plt.ylabel ('frequency', fontsize=40)
                plt.legend(loc='upper right', prop={'size': 30})
                plt.savefig(path_to_output_plot_folder + plotname)

def plot_qflag_obs_hist(name_stat, date_in, date_fin, obs_file, key_obs_file, qflag_obs_ts, path_to_output_plot_folder):
                plotname = name_stat + '_' + date_in + '_' + date_fin + '_qflag_obs_histogram.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=16)
                plt.title('Observation Quality Flag Frequency Distribution: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n --- Time Period: '+ date_in + '-' + date_fin, fontsize=26)
                bins = np.linspace(np.amin(qflag_obs_ts[name_stat][~np.isnan(qflag_obs_ts[name_stat])]), np.amax(qflag_obs_ts[name_stat][~np.isnan(qflag_obs_ts[name_stat])]),10)
                plt.hist(qflag_obs_ts[name_stat][~np.isnan(qflag_obs_ts[name_stat])], bins, label='qflag_obs_hist')
                ax.tick_params(axis='both', labelsize=13)
                plt.xlabel ('quality flag', fontsize=40)
                plt.ylabel ('frequency', fontsize=40)
                plt.legend(loc='upper right', prop={'size': 30})
                plt.savefig(path_to_output_plot_folder + plotname)
                plt.clf()

def plot_mod_obs_hist(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, path_to_output_plot_folder):
                plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + '_mod_obs_histograms.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=16)
                plt.title('Surface (3m) Current Velocity Frequency Distribution: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n --- Time Period: '+ date_in + '-' + date_fin, fontsize=26)
                hist(vel_mod_ts[name_stat], bins="scott", ax=ax, histtype='stepfilled',alpha=0.5, density=True, label='model_hist')
                hist(vel_obs_ts[name_stat], bins="scott", ax=ax, histtype='stepfilled',alpha=0.5, density=True, label='observation')
                ax.tick_params(axis='both', labelsize=13)
                plt.xlabel ('velocity [m/s]', fontsize=40)
                plt.ylabel ('frequency', fontsize=40)
                plt.legend(loc='upper right', prop={'size': 30})
                plt.savefig(path_to_output_plot_folder + plotname)
                plt.clf()

def plot_mod_obs_ECDF(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, path_to_output_plot_folder):

                plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ECDF.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=16)
                plt.grid()
                plt.title('Surface (3m) Current Velocity ECDF: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n Period: '+ date_in + ' - ' + date_fin, fontsize=29)
                ax.tick_params(axis='both', labelsize=26)
                plt.xlabel ('velocity [m/s]', fontsize=40)
                plt.ylabel ('ECDF', fontsize=40)
                ecdf_obs = ECDF(np.array(vel_obs_ts[name_stat]))
                ecdf_mod = ECDF(np.array(vel_mod_ts[name_stat]))
                plt.axhline(y=0.5, color='black', linestyle="dashed")
                plt.plot(ecdf_mod.x,ecdf_mod.y,label="model", linewidth=4)
                plt.plot(ecdf_obs.x,ecdf_obs.y,label="observation", linewidth=4)
                plt.legend( loc='lower right', prop={'size': 40})
                plt.text(0.17, 0.89, name_exp[0], weight='bold',transform=plt.gcf().transFigure,fontsize=22)
                plt.savefig(path_to_output_plot_folder + plotname)
                plt.clf()

def plot_bias_rmse_ts(date_in, date_fin, time_res, timerange, bias_ts, rmsd_ts, statistics_array, time_res_xaxis, name_exp, path_to_output_plot_folder):
            plotname = date_in + '_' + date_fin + '_' + time_res + '_bias_rmse_ts.png'
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel('Date', fontsize=40)
            ax1.set_ylabel('BIAS [m/s]', fontsize=40, color='darkblue')
            plt.rc('font', size=8)
            plt.title('Surface (3m) Current Velocity BIAS and RMSD -ALL: \n Period: '+ date_in + '-' + date_fin, fontsize=29)
            lns1 = ax1.plot(timerange,list(bias_ts.values()),label = 'BIAS: {} m/s'.format(statistics_array[0]), linewidth=3, color='darkblue')
            ax1.axhline(y=0, color='k', linestyle='--')
            ax1.tick_params(axis='y', labelsize=26, colors='darkblue')
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('RMSD [m/s]', fontsize=40, color='orange')
            lns2 = ax2.plot(timerange,np.sqrt(list(rmsd_ts.values())),color=color,label = 'RMSD: {} m/s'.format(statistics_array[1]), linewidth=4)
            ax2.tick_params(axis='y', labelsize=26, colors='orange')
            if time_res_xaxis[1]=='w':
               ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
            if time_res_xaxis[1]=='m':
               ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
            if time_res_xaxis[1]=='y':
               ax1.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_xaxis[0])))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            fig.autofmt_xdate()
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)
            ax1.grid(linestyle='-')
            plt.text(0.17, 0.89, name_exp[0], weight='bold',transform=plt.gcf().transFigure,fontsize=22)
            nticks = 8
            ax1.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
            ax2.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
            ax1.grid('on')
            lns=lns1+lns2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper left',  prop={'size': 20}, framealpha=0.2)
            plt.savefig(path_to_output_plot_folder + plotname, dpi=300, bbox_inches = "tight")
            plt.clf()

def plot_mod_obs_ECDF_total(date_in, date_fin, time_res, mod_array, obs_array, path_to_output_plot_folder):
            plotname = date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ECDF.png'
            fig = plt.figure(figsize=(18,12))
            ax = fig.add_subplot(111)
            plt.rc('font', size=16)
            plt.grid()
            plt.title('Surface (3m) Current Velocity ECDF -ALL:\n Period: '+ date_in + '-' + date_fin, fontsize=29)
            plt.xlabel ('velocity [m/s]', fontsize=40)
            plt.ylabel ('ECDF', fontsize=40)
            ecdf_obs = ECDF(obs_array)
            ecdf_mod = ECDF(mod_array)
            plt.axhline(y=0.5, color='black', linestyle="dashed")
            plt.plot(ecdf_mod.x,ecdf_mod.y,label="model velocity [m/s]", linewidth=4)
            plt.plot(ecdf_obs.x,ecdf_obs.y,label="observation velocity [m/s]", linewidth=4)
            ax.tick_params(axis='both', labelsize=26)
            plt.legend( loc='lower right' , prop={'size': 40})
            plt.text(0.17, 0.89, name_exp[0], weight='bold',transform=plt.gcf().transFigure,fontsize=22)
            plt.savefig(path_to_output_plot_folder + plotname)
            plt.clf()

def plot_mod_obs_ts_diff_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, num_exp, vel_mod_ts, vel_obs_ts, name_exp, timerange, time_res_xaxis, path_to_output_plot_folder):

                plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ts_difference_comparison.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=24)
                plt.title('Surface (3m) Current Velocity BIAS: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: '+ date_in + '-' + date_fin, fontsize=29)
                for exp in range(num_exp):
                    mean_vel_bias = round(np.nanmean(np.array(vel_mod_ts[exp][name_stat]-np.array(vel_obs_ts[name_stat]))),2)
                    mean_vel_rmsd = round(math.sqrt(np.nanmean((np.array(vel_mod_ts[exp][name_stat]-np.array(vel_obs_ts[name_stat]))**2))),2)
                    plt.plot(timerange,vel_mod_ts[exp][name_stat]-vel_obs_ts[name_stat],label = name_exp[exp] + ' (BIAS: '+str(mean_vel_bias)+' m/s)', linewidth=3)
                    plt.plot([], [], ' ', label = name_exp[exp] + ' (RMSD: ' + str(mean_vel_rmsd)+' m/s)')
                plt.grid()
                ax.tick_params(axis='both', labelsize=26)
                if time_res_xaxis[1]=='w':
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
                if time_res_xaxis[1]=='m':
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
                if time_res_xaxis[1]=='y':
                    ax.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_xaxis[0])))

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
                fig.autofmt_xdate()
                plt.ylabel('Velocity Difference [m/s]', fontsize=40)
                plt.xlabel('Date', fontsize=40)
                plt.legend(prop={'size': 20}, framealpha=0.2)
                plt.savefig(path_to_output_plot_folder + plotname)
                plt.clf()

def plot_mod_obs_ts_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, num_exp, vel_mod_ts, vel_obs_ts, name_exp, timerange, time_res_xaxis, path_to_output_plot_folder):
                plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ts_comparison.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=24)
                plt.title('Surface (3m) Current Velocity: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: '+ date_in + '-' + date_fin, fontsize=29)
                for exp in range(num_exp):
                    mean_vel_mod = round(np.nanmean(np.array(vel_mod_ts[exp][name_stat])),2)
                    plt.plot(timerange,vel_mod_ts[exp][name_stat],label = name_exp[exp] + ' : '+str(mean_vel_mod)+' m/s', linewidth=2)
                mean_vel_obs = round(np.nanmean(np.array(vel_obs_ts[name_stat])),2)
                plt.plot(timerange,vel_obs_ts[name_stat],label = 'Observation : '+str(mean_vel_obs)+' m/s', linewidth=2)
                plt.grid()
                ax.tick_params(axis='both', labelsize=26)
                if time_res_xaxis[1]=='w':
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
                if time_res_xaxis[1]=='m':
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
                if time_res_xaxis[1]=='y':
                    ax.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_xaxis[0])))

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
                fig.autofmt_xdate()
                plt.ylabel('Velocity [m/s]', fontsize=40)
                plt.xlabel('Date', fontsize=40)
                plt.legend(prop={'size': 30}, framealpha=0.2)
                plt.savefig(path_to_output_plot_folder + plotname)
                plt.clf()

def plot_mod_obs_ECDF_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, num_exp, vel_mod_ts, vel_obs_ts, name_exp, path_to_output_plot_folder):
                plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ECDF_comparison.png'
                fig = plt.figure(figsize=(18,12))
                ax = fig.add_subplot(111)
                plt.rc('font', size=16)
                plt.grid ()
                plt.title('Surface (3m) Current Velocity ECDF: '+ name_stat + ' (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n Period: '+ date_in + ' - ' + date_fin, fontsize=29)
                ax.tick_params(axis='both', labelsize=26)
                plt.xlabel ('velocity [m/s]', fontsize=40)
                plt.ylabel ('ECDF', fontsize=40)
                ecdf_obs = ECDF(np.array(vel_obs_ts[name_stat]))
                plt.axhline(y=0.5, color='black', linestyle="dashed")
                for exp in range(num_exp):
                    ecdf_mod = ECDF(np.array(vel_mod_ts[exp][name_stat]))
                    plt.plot(ecdf_mod.x,ecdf_mod.y,label=name_exp[exp], linewidth=4)
                plt.plot(ecdf_obs.x,ecdf_obs.y,label="observation", linewidth=4)
                plt.legend( loc='lower right', prop={'size': 40})
                plt.savefig(path_to_output_plot_folder + plotname)
                plt.clf()

def plot_tot_mod_obs_ECDF_comparison(date_in, date_fin, time_res, num_exp, vel_mod_ts, vel_obs_ts, name_exp, path_to_output_plot_folder):
            plotname = date_in + '_' + date_fin + '_' + time_res + '_mod_obs_ECDF_comparison.png'
            fig = plt.figure(figsize=(18,12))
            ax = fig.add_subplot(111)
            plt.rc('font', size=16)
            plt.grid()
            plt.title('Surface (3m) Current Velocity ECDF -ALL:\n Period: '+ date_in + '-' + date_fin, fontsize=29)
            plt.xlabel ('velocity [m/s]', fontsize=40)
            plt.ylabel ('ECDF', fontsize=40)
            plt.axhline(y=0.5, color='black', linestyle="dashed")

            for exp in range(num_exp):
                mod_array = np.array([])
                if exp==0:
                    obs_array = np.array([])
                for name_stat in vel_mod_ts[0].keys():

                    mod_array = np.concatenate([mod_array, np.array(vel_mod_ts[exp][name_stat])])
                    if exp==0:
                        obs_array = np.concatenate([obs_array, np.array(vel_obs_ts[name_stat])])

                ecdf_obs = ECDF(obs_array)
                ecdf_mod = ECDF(mod_array)

                plt.plot(ecdf_mod.x,ecdf_mod.y,label=name_exp[exp], linewidth=4)

            plt.plot(ecdf_obs.x,ecdf_obs.y,label="observation", linewidth=4)
            ax.tick_params(axis='both', labelsize=26)
            plt.legend( loc='lower right' , prop={'size': 40})
            plt.savefig(path_to_output_plot_folder + plotname)
            plt.clf()

def plot_bias_ts_comparison(date_in, date_fin, time_res, num_exp, timerange, name_exp, bias_ts, time_res_xaxis, path_to_output_plot_folder):

            plotname = date_in + '_' + date_fin + '_' + time_res + '_bias_ts_comparison.png'
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel('Date', fontsize=40)
            ax1.set_ylabel('BIAS [m/s]', fontsize=40)
            plt.rc('font', size=8)
            plt.title('Surface (3m) Current Velocity BIAS -ALL: \n Period: '+ date_in + '-' + date_fin, fontsize=29)
            for exp in range(num_exp):
                ax1.plot(timerange,list(bias_ts[exp].values()),label = name_exp[exp]+' : {} m/s'.format(round(np.nanmean(np.array(list(bias_ts[exp].values()))),2)), linewidth=3)
            ax1.axhline(y=0, color='k', linestyle='--')
            ax1.tick_params(axis='y', labelsize=26)
            if time_res_xaxis[1]=='w':
               ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
            if time_res_xaxis[1]=='m':
               ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
            if time_res_xaxis[1]=='y':
               ax1.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_xaxis[0])))

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            fig.autofmt_xdate()
            ax1.tick_params(axis='x', labelsize=20)
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)
            ax1.grid(linestyle='-')
            nticks = 8
            ax1.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
            ax1.grid('on')

            ax1.legend(loc='upper left',  prop={'size': 40}, framealpha=0.2)
            plt.savefig(path_to_output_plot_folder + plotname, dpi=300, bbox_inches = "tight")
            plt.clf()

def plot_rmse_ts_comparison(date_in, date_fin, time_res, num_exp, timerange, name_exp, rmsd_ts, time_res_xaxis, path_to_output_plot_folder):
            plotname = date_in + '_' + date_fin + '_' + time_res + '_rmse_ts_comparison.png'
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel('Date', fontsize=40)
            ax1.set_ylabel('RMSD [m/s]', fontsize=40)
            plt.rc('font', size=8)
            plt.title('Surface (3m) Current Velocity RMSD -ALL: \n Period: '+ date_in + '-' + date_fin, fontsize=29)
            for exp in range(num_exp):
                ax1.plot(timerange,np.sqrt(list(rmsd_ts[exp].values())),label = name_exp[exp] + ' : {} m/s'.format(round(math.sqrt(np.nanmean(np.array(list(rmsd_ts[exp].values())))),2)), linewidth=3)
            ax1.tick_params(axis='y', labelsize=26)

            if time_res_xaxis[1]=='w':
               ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
            if time_res_xaxis[1]=='m':
               ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
            if time_res_xaxis[1]=='y':
               ax1.xaxis.set_major_locator(mdates.YearLocator(interval=int(time_res_xaxis[0])))

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            fig.autofmt_xdate()
            ax1.tick_params(axis='x', labelsize=20)
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)
            ax1.grid(linestyle='-')
            nticks = 8
            ax1.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
            ax1.grid('on')

            ax1.legend(loc='upper left',  prop={'size': 40}, framealpha=0.2)
            plt.savefig(path_to_output_plot_folder + plotname, dpi=300, bbox_inches = "tight")
            plt.clf()

if __name__ == "__main__":

    argv=sys.argv
    print("argv: ", argv)
    path_to_obs_file=argv[1]
    path_to_obs_ts_folder=argv[2]
    path_to_output_plot_folder=argv[3]
    date_in=argv[4]
    date_fin=argv[5]
    time_res=argv[6]
    time_res_xaxis=argv[7]
    num_exp=int(argv[8])
    work_dir_plot=argv[9]
    path_to_mod_ts_folder=argv[10:10+num_exp]
    name_exp=argv[10+num_exp:10+2*num_exp]

    os.makedirs(path_to_output_plot_folder,  exist_ok=True)

    start_date = date(int(date_in[0:4]),int(date_in[4:6]) , int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]),int(date_fin[4:6]) , int(date_fin[6:8]))
    timerange = pd.date_range(start_date, end_date)

    obs_file = {}
    with open(path_to_obs_file) as f:
        reader = csv.reader(f,delimiter=';')
        next(reader)
        for row in reader:
            print(row)
            array_name = np.array([row[4], row[5], row[6]])
            boolArr = np.where(array_name != "_")
            obs_file[array_name[boolArr][0]] = {'lat': row[0], 'lon': row[1], 'depth': '3'}

    if num_exp==1:
        onlyfiles_mod = [f for f in sorted(listdir(path_to_mod_ts_folder[0])) if isfile(join(path_to_mod_ts_folder[0], f))]
        onlyfiles_obs = [f for f in sorted(listdir(path_to_obs_ts_folder)) if isfile(join(path_to_obs_ts_folder, f))]

        statistics={}
        vel_mod_ts={}
        vel_obs_ts={}
        depth_obs_ts={}
        qflag_obs_ts={}
        for filename_mod, filename_obs in zip(onlyfiles_mod, onlyfiles_obs):
            print(filename_mod)
            print(filename_obs)
            splitted_name = np.array(filename_mod.split("_"))
            start_date_index = np.argwhere(splitted_name==date_in)
            print(start_date_index)
            name_station_splitted = splitted_name[0:start_date_index[0][0]]
            name_station = '_'.join(name_station_splitted)
            mod_ts = NC.Dataset(path_to_mod_ts_folder[0] + filename_mod,'r')
            vel_mod_ts[name_station] = ma.getdata(mod_ts.groups['current velocity time series'].variables['Current Velocity'][:])
            obs_ts = NC.Dataset(path_to_obs_ts_folder + filename_obs,'r')
            vel_obs_ts[name_station] = ma.getdata(obs_ts.groups['current velocity time series'].variables['Current Velocity'][:])
            depth_obs_ts[name_station] = ma.getdata(obs_ts.groups['depth time series'].variables['Depth'][:])
            qflag_obs_ts[name_station] = ma.getdata(obs_ts.groups['qflag time series'].variables['Qflag'][:])

        for key_obs_file, name_stat in zip(sorted(obs_file.keys()),vel_mod_ts.keys()):
            
            print(name_stat)
            plot_mod_obs_ts_diff(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, time_res_xaxis, path_to_output_plot_folder)

            tot_mean_stat=plot_mod_obs_ts(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, time_res_xaxis, path_to_output_plot_folder)

            #plot_depth_obs_hist(name_stat, date_in, date_fin, obs_file, key_obs_file, depth_obs_ts, path_to_output_plot_folder)

            #plot_qflag_obs_hist(name_stat, date_in, date_fin, obs_file, key_obs_file, qflag_obs_ts, path_to_output_plot_folder)

            #plot_mod_obs_hist(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, path_to_output_plot_folder)

            plot_mod_obs_ECDF(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, path_to_output_plot_folder)

            plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res +  '_qqPlot.png'
            title = 'Surface (3m) Current Velocity ' + name_stat + '\n (' + obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') Period: ' + date_in + ' - ' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            print(vel_mod_ts[name_stat])
            print(vel_obs_ts[name_stat])
            statistics_array=scatterPlot(np.array(vel_mod_ts[name_stat]),np.array(vel_obs_ts[name_stat]),path_to_output_plot_folder + plotname,name_exp,title=title,xlabel=xlabel,ylabel=ylabel)
            row_stat = tot_mean_stat + statistics_array
            statistics[name_stat] = row_stat   


        bias_ts={}
        diff_q_ts={}
        rmsd_ts={}

        for day in daterange(start_date,end_date):
            timetag = day.strftime("%Y%m%d")
            bias_ts[timetag] = 0
            diff_q_ts[timetag] = 0
            rmsd_ts[timetag] = 0

        for i, day in enumerate(daterange(start_date,end_date)):
            timetag = day.strftime("%Y%m%d")
            
            for name_stat in vel_mod_ts.keys():
                if np.isnan(vel_obs_ts[name_stat][i]):
                    continue
                else:
                    bias = vel_mod_ts[name_stat][i] - vel_obs_ts[name_stat][i]
                    diff_q = (vel_mod_ts[name_stat][i]- vel_obs_ts[name_stat][i])**2
                    bias_ts[timetag] = bias_ts[timetag] + bias
                    diff_q_ts[timetag] = diff_q_ts[timetag] + diff_q
            rmsd_ts[timetag] = diff_q_ts[timetag]/len(vel_mod_ts.keys())
            bias_ts[timetag] = bias_ts[timetag]/len(vel_mod_ts.keys())
        
        mod_array = np.array([])
        obs_array = np.array([])
        for name_stat in vel_mod_ts.keys():
            mod_array = np.concatenate([mod_array, np.array(vel_mod_ts[name_stat])])
            obs_array = np.concatenate([obs_array, np.array(vel_obs_ts[name_stat])])

        tot_mean_mod=round(np.nanmean(mod_array),2)
        tot_mean_obs=round(np.nanmean(obs_array),2)
        mean_all=[tot_mean_mod,tot_mean_obs]

        plotname = date_in + '_' + date_fin + '_' + time_res +  '_qqPlot.png'
        title = 'Surface (3m) Current Velocity -ALL \n Period: ' + date_in + '-' + date_fin
        xlabel = 'Observation Current Velocity [m/s]'
        ylabel = 'Model Current Velocity [m/s]'
        statistics_array=scatterPlot(mod_array,obs_array,path_to_output_plot_folder + plotname,name_exp,title=title,xlabel=xlabel,ylabel=ylabel)
        row_all = mean_all + statistics_array
        statistics["ALL BUOYS"] = row_all
       
        plot_bias_rmse_ts(date_in, date_fin, time_res, timerange, bias_ts, rmsd_ts, statistics_array, time_res_xaxis, name_exp, path_to_output_plot_folder)

        plot_mod_obs_ECDF_total(date_in, date_fin, time_res, mod_array, obs_array, path_to_output_plot_folder)

        lon={}
        lat={}
        mapping(obs_file,lat,lon,work_dir_plot)

        a_file = open(work_dir_plot+"statistics_" + name_exp[0] + "_" + date_in + "_" + date_fin + ".csv", "w")
        writer = csv.writer(a_file)
        writer.writerow(["name_station", "mean_mod", "mean_obs", "bias","rmse","si","corr","stderr","number_of_obs"])
        for key, value in statistics.items():
            array = [key] + value
            print(array)
            writer.writerow(array)
        a_file.close()

    if num_exp > 1:
        vel_mod_ts={ }     
        for exp in range(num_exp):
            onlyfiles_mod = [f for f in sorted(listdir(path_to_mod_ts_folder[exp])) if isfile(join(path_to_mod_ts_folder[exp], f))]
            onlyfiles_obs = [f for f in sorted(listdir(path_to_obs_ts_folder)) if isfile(join(path_to_obs_ts_folder, f))]

            vel_mod_ts[exp]={}
            vel_obs_ts={}
            for filename_mod, filename_obs in zip(onlyfiles_mod, onlyfiles_obs):
                splitted_name = np.array(filename_mod.split("_"))
                start_date_index = np.argwhere(splitted_name==date_in)
                name_station_splitted = splitted_name[0:start_date_index[0][0]]
                name_station = '_'.join(name_station_splitted)
                mod_ts = NC.Dataset(path_to_mod_ts_folder[exp] + filename_mod,'r')
                vel_mod_ts[exp][name_station] = ma.getdata(mod_ts.groups['current velocity time series'].variables['Current Velocity'][:])
                obs_ts = NC.Dataset(path_to_obs_ts_folder + filename_obs,'r')
                vel_obs_ts[name_station] = ma.getdata(obs_ts.groups['current velocity time series'].variables['Current Velocity'][:])

        for key_obs_file, name_stat in zip(sorted(obs_file.keys()),vel_mod_ts[0].keys()): 

            plot_mod_obs_ts_diff_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, num_exp, vel_mod_ts, vel_obs_ts, timerange, time_res_xaxis, path_to_output_plot_folder)
            plot_mod_obs_ts_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, num_exp, vel_mod_ts, vel_obs_ts, name_exp, timerange, time_res_xaxis, path_to_output_plot_folder)
            plot_mod_obs_ECDF_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, num_exp, vel_mod_ts, vel_obs_ts, name_exp, path_to_output_plot_folder)

        plot_tot_mod_obs_ECDF_comparison(date_in, date_fin, time_res, num_exp, vel_mod_ts, vel_obs_ts, name_exp, path_to_output_plot_folder)

        bias_ts={ }
        diff_q_ts={ }
        rmsd_ts={ }

        for exp in range(num_exp):
            bias_ts[exp] = {}
            diff_q_ts[exp] = {}
            rmsd_ts[exp] = {}

            for day in daterange(start_date,end_date):
                timetag = day.strftime("%Y%m%d")
                bias_ts[exp][timetag] = 0
                diff_q_ts[exp][timetag] = 0
                rmsd_ts[exp][timetag] = 0


            for i, day in enumerate(daterange(start_date,end_date)):
                timetag = day.strftime("%Y%m%d")
                print(timetag)
                for name_stat in vel_mod_ts[0].keys():
                    if np.isnan(vel_obs_ts[name_stat][i]):
                        print("nan beccato!")
                        continue
                    else:
                        bias = vel_mod_ts[exp][name_stat][i] - vel_obs_ts[name_stat][i]
                        diff_q = (vel_mod_ts[exp][name_stat][i]- vel_obs_ts[name_stat][i])**2
                        bias_ts[exp][timetag] = bias_ts[exp][timetag] + bias
                        diff_q_ts[exp][timetag] = diff_q_ts[exp][timetag] + diff_q
                rmsd_ts[exp][timetag] = diff_q_ts[exp][timetag]/len(vel_mod_ts[exp].keys())
                bias_ts[exp][timetag] = bias_ts[exp][timetag]/len(vel_mod_ts[exp].keys())
        print("rmsd_ts: ", rmsd_ts[0])

        plot_bias_ts_comparison(date_in, date_fin, time_res, num_exp, timerange, name_exp, bias_ts, time_res_xaxis, path_to_output_plot_folder)
        plot_rmse_ts_comparison(date_in, date_fin, time_res, num_exp, timerange, name_exp, rmsd_ts, time_res_xaxis, path_to_output_plot_folder)
