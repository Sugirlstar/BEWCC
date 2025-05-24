import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import cmocean
import math
import time
import geopandas
from HYJfunction import *

from netCDF4 import Dataset
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.colors
import os
import cartopy
from cartopy import crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from scipy import ndimage
from multiprocessing import Pool, Manager
import cartopy.feature as cfeature
from scipy.ndimage import convolve
from scipy.signal import detrend
import pickle
import xarray as xr
import regionmask
from matplotlib.patches import Polygon
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
from scipy import stats
from collections import defaultdict
from scipy.ndimage import label
from scipy.interpolate import interp2d
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# # Data Preparation -----------------
# # 1. blocking index (3d: time, lat, lon) - define blocking -1dgree!
# # Clusters
# with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily", "rb") as fp:
#     Blocking_diversity_label = pickle.load(fp)    
# Blocking_allType_label = [item for sublist in Blocking_diversity_label for item in sublist]
# print(len(Blocking_allType_label))
# # the length of three types:
# Blktypearr = np.array([0]*len(Blocking_diversity_label[0]) + [1]*len(Blocking_diversity_label[1]) + [2]*len(Blocking_diversity_label[2]))
# print(len(Blocking_diversity_label[0]))
# print(len(Blocking_diversity_label[1]))
# print(len(Blocking_diversity_label[2]))

# # label structure: 2d array, True/False label for blocked or not, at each location
# with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily", "rb") as fp:
#     Blocking_diversity_date = pickle.load(fp)      
# Blocking_allType_date = [item for sublist in Blocking_diversity_date for item in sublist]

# # Peaking Centers
# with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_date_daily", "rb") as fp:
#     Blocking_diversity_peaking_date = pickle.load(fp)
# with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lon_daily", "rb") as fp:
#     Blocking_diversity_peaking_lon = pickle.load(fp)
# with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_peaking_lat_daily", "rb") as fp:
#     Blocking_diversity_peaking_lat = pickle.load(fp)
# Blocking_allType_peaking_date = [item for sublist in Blocking_diversity_peaking_date for item in sublist]
# Blocking_allType_peaking_lon = [item for sublist in Blocking_diversity_peaking_lon for item in sublist]
# Blocking_allType_peaking_lat = [item for sublist in Blocking_diversity_peaking_lat for item in sublist]

# # # find the 2021 Pacific Northeast block, and the 2010 Russian heatwave
# # # PN21 - time: 20210622-20210630, lat: 40-60, lon: 200-240
# # # RH10 - time: 20100723-20100814, lat: 48-72, lon: 20-60
# # event_dates = np.array(Blocking_allType_peaking_date)  # 类型为 pd.Timestamp
# # event_lats = np.array(Blocking_allType_peaking_lat)
# # event_lons = np.array(Blocking_allType_peaking_lon)
# # # --- PN21 ---
# # pn21_mask = (
# #     (event_dates >= pd.Timestamp("2021-06-22")) & 
# #     (event_dates <= pd.Timestamp("2021-06-30")) &
# #     (event_lats >= 40) & (event_lats <= 60) &
# #     (event_lons >= 200) & (event_lons <= 240)
# # )
# # pn21_indices = np.where(pn21_mask)[0]  
# # print(pn21_indices)
# # # --- RH10 ---
# # rh10_mask = (
# #     (event_dates >= pd.Timestamp("2010-07-23")) & 
# #     (event_dates <= pd.Timestamp("2010-08-14")) &
# #     (event_lats >= 48) & (event_lats <= 72) &
# #     (event_lons >= 20) & (event_lons <= 60)
# # )
# # rh10_indices = np.where(rh10_mask)[0]
# # print(rh10_indices)

# # lat, lon, time Alignment
# lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
# lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
# lat_mid = int(len(lat)/2) + 1 #90
# Blklat = lat[0:lat_mid-1] # NH latitudes: 90-1
# Blklon = lon
# print(Blklat)
# print(len(Blklat))
# print(Blklon)
# # time for blocking indices 
# Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
# Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
# timestamp = list(Date0['date'])
# timestamparr = np.array(timestamp)

# # 2. temperature (T) anomaly (3d: time, lat, lon) - 0.25drgee!
# ncfile = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Tsurface_1980_2021_20_70N_daily.nc') # 19500101-20211231
# t_k = np.array(ncfile['t'].squeeze())
# print(t_k.shape)
# lats_TS = np.array(ncfile['latitude']) # 70-20, by 0.25
# lats_TS = np.flip(lats_TS) # Make it 20-70
# print(lats_TS) # 20-70, by 0.25
# t_k = np.flip(t_k, axis=1) # Make it 20-70
# lons_TS = np.array(ncfile['longitude']) # 0-359.75 by 0.25
# ncfile.close()
# ts = t_k-273.15
# distributionPlot(ts[0,:,:],lons_TS,lats_TS,'ERA5_ts_d1')

# # calculate the anomalies
# timei = []
# doy_list = []  
# start_date = dt.datetime(1980, 1, 1)
# end_date = dt.datetime(2021, 12, 31)
# for ordinal in range(start_date.toordinal(), end_date.toordinal() + 1):
#     current_date = dt.datetime.fromordinal(ordinal)
#     date_str = current_date.strftime('%m%d') + str(current_date.year).zfill(4)
#     timei.append(date_str)
#     doy_list.append(current_date.timetuple().tm_yday)
# doy = doy_list # get DOY
# unique_doy = np.unique(doy)  # get the only DOY
# climatology = np.empty((len(unique_doy), ts.shape[1], ts.shape[2]))
# Tanomly = np.empty_like(ts)
# for i, day in enumerate(unique_doy):
#     day_indices = np.where(doy == day)[0]
#     climatology[i,:,:] = np.nanmean(ts[day_indices,:,:], axis=0)
#     Tanomly[day_indices,:,:] = ts[day_indices,:,:] - climatology[i,:,:]
# np.save('/scratch/bell/hu1029/Data/processed/ERA5_Tsurface_1980_2021_20_70N_dailyAnomaly.npy', Tanomly)
# distributionPlot(Tanomly[0,:,:],lons_TS,lats_TS,'ERA5_TSanomaly_d1')
# Tanomly = np.load('/scratch/bell/hu1029/Data/processed/ERA5_Tsurface_1980_2021_20_70N_dailyAnomaly.npy')
# # time for Tanomaly
# Datestamp_TS = pd.date_range(start="1980-01-01", end="2021-12-31")
# Date0_TS = pd.DataFrame({'date': pd.to_datetime(Datestamp_TS)})
# timestamp_TS = list(Date0_TS['date'])
# print(len(timestamp_TS))

# # 3. TCWV anomaly (3d: time, lat, lon)
# ncfile = xr.open_dataset('/scratch/bell/zhao1550/data_raw/precip/total_precip_NH_1980_2021_r360x181.nc') # 
# tcwv = np.array(ncfile['var228'].squeeze())
# lats_TS = np.array(ncfile['lat']) # 70-20, by 0.25
# lats_TS = np.flip(lats_TS) # Make it 20-70
# midlat = np.where((Blklat >= 20) & (Blklat <= 70))[0]
# lats_TS = lats_TS[midlat]
# tcwv = np.flip(tcwv, axis=1) # Make it 20-70
# tcwv = tcwv[:, midlat,:]
# print(lats_TS) # 20-70, by 0.25
# lons_TS = np.array(ncfile['lon']) # 0-359.75 by 0.25
# ncfile.close()
# distributionPlot(tcwv[0,:,:],lons_TS,lats_TS,'ERA5_Prep_d1')
# print(tcwv.shape)
# TCWVanomly = tcwv

# # # # 4. geopotential height anomaly (3d: time, lat, lon; event-based, selected from blocking index)

# # 5. get the average T, P anomaly based on each blocking event, on each longitude (2d: time, lon)
# BlkpeakLonlist = []
# BlkTanom = []
# BlkPanom = []
# BlkDuration = []
# targetEeventid = []
# BlkTypeid = []

# for i in range(len(Blocking_allType_date)):

#     eventDates = Blocking_allType_date[i] # a list of dates
#     eventLabelArr = Blocking_allType_label[i] # a list of 2d arrays
#     eventPeakingDate = Blocking_allType_peaking_date[i] # a value
#     eventPeakingLon = Blocking_allType_peaking_lon[i] # a value
#     eventPeakingLat = Blocking_allType_peaking_lat[i] # a value

#     # make sure the event is in the target region and time
#     if 20 <= eventPeakingLat <= 70:
#         if all(d in Date0_TS['date'].values for d in eventDates):

#             # 01 id, lon, duration
#             # targeteventid
#             targetEeventid.append(i)
#             # longitude value:
#             BlkpeakLonlist.append(eventPeakingLon)
#             # duration:
#             BlkDuration.append(len(eventDates))
#             # type id:
#             BlkTypeid.append(Blktypearr[i])
#             # use the label to crop the Tanomly array
            
#             # 02 multiply the label (1/nan)
#             # grid alignment
#             blklabel = np.array(eventLabelArr)
#             blklabel[np.where(blklabel == 0)] = np.nan # make the label 1/nan
#             midlat = np.where((Blklat >= 20) & (Blklat <= 70))[0]
#             blklabel = blklabel[:, midlat,:] # get the latitudes 20-70N, but value order is 70-20N
#             blklabel = np.flip(blklabel, axis=1) # make the value along increased lat, from 20-70N
#             # make it (1dg) align with the TS and P anomaly array (0.25dg)
#             blklabel025dg = np.repeat(np.repeat(blklabel, 4, axis=1), 4, axis=2)
#             blklabel025dg = blklabel025dg[:,:-3,:]
            
#             # calculate the event averaged value
#             timeindex = [timestamp_TS.index(val) for val in eventDates] # get the time index
#             EventTSarray = Tanomly[timeindex,:,:] # get the Tanomly
#             EventTSLabeled = EventTSarray * blklabel025dg # get the Tanomly
#             EvtTSanom = np.nanmean(EventTSLabeled) # get the event averaged Tanomly
#             BlkTanom.append(EvtTSanom)

#             # get the Panomly
#             EventPSarrat = TCWVanomly[timeindex,:,:] # get the TCWVanomly
#             EventPSLabeled = EventPSarrat * blklabel
#             EvtPSanom = np.nanmean(EventPSLabeled)
#             BlkPanom.append(EvtPSanom)
#             # BlkPanom.append(...)

# print(BlkpeakLonlist,flush=True)   
# print(len(targetEeventid), flush=True)

# # Plot the figure -----------------
# # left panel: scatter plot of blocking event index point, x-axis: longitude, y-axis: T anomaly, point color: P anomaly
# # right panel: (b-e) four selected events: x-y: lat-lon; countours: Z500 anomaly; color shading: T anomaly; hatches: P anomaly

# # Panel (a)
# BlkpeakLonlist = np.array(BlkpeakLonlist)
# BlkTanom = np.array(BlkTanom)
# BlkPanom = np.array(BlkPanom)
# BlkDuration = np.array(BlkDuration)
# BlkTypeid = np.array(BlkTypeid)
# duration_min = np.nanmin(BlkDuration)
# duration_max = np.nanmax(BlkDuration)
# print("Duration min:", duration_min, flush=True)
# print("Duration max:", duration_max, flush=True)
# specialEeventList = np.full_like(BlkTanom,np.nan)
# targetEeventid = np.array(targetEeventid)
# # pn21_point_loc = np.where(np.isin(targetEeventid, pn21_indices))
# # rh10_point_loc = np.where(np.isin(targetEeventid, rh10_indices))

# np.savez('/scratch/bell/hu1029/blocking_event_data_precprawvalue.npz',
#          BlkTanom=BlkTanom,
#          BlkPanom=BlkPanom,
#          BlkDuration=BlkDuration,
#          BlkTypeid=BlkTypeid,
#          targetEeventid=targetEeventid,
#          BlkpeakLonlist=BlkpeakLonlist)

# load 
data = np.load('/scratch/bell/hu1029/blocking_event_data_precp5dmovingMean_Intensity.npz', allow_pickle=True)
BlkTanom = data['BlkTanom']
BlkPanom = data['BlkPanom']
BlkDuration = data['BlkDuration']
BlkTypeid = data['BlkTypeid']
BlkpeakLonlist = data['BlkpeakLonlist']
BlkTimeid = data['BlkTimeid']
BlkIntensity = data['BlkIntensity']

# transfer BlkpeakLonlist
# BlkLon_plot = np.where(BlkpeakLonlist < 270, BlkpeakLonlist + 360, BlkpeakLonlist)
# scatter plot
BlkTimeid_dt = pd.to_datetime(BlkTimeid)
is_JJA = BlkTimeid_dt.month.isin([6, 7, 8])
is_DJF = BlkTimeid_dt.month.isin([12, 1, 2])
JJA_top100_idx = np.argsort(BlkIntensity[is_JJA])[-100:]  
DJF_top100_idx = np.argsort(BlkIntensity[is_DJF])[-100:]  
JJA_global_idx = np.where(is_JJA)[0][JJA_top100_idx]
DJF_global_idx = np.where(is_DJF)[0][DJF_top100_idx]

combined_idx = np.concatenate([JJA_global_idx, DJF_global_idx])
BlkTimeid_dt_sub = BlkTimeid_dt[combined_idx]
is_early = (BlkTimeid_dt_sub.year <= 1999)
is_late  = (BlkTimeid_dt_sub.year >= 2000)


fig, ax = plt.subplots(figsize=(7, 6))

duration_min = np.nanmin(BlkpeakLonlist)
duration_max = np.nanmax(BlkpeakLonlist)
vabs = np.nanmax(np.abs(BlkPanom))  

ax.scatter(
    BlkPanom[combined_idx][is_early], BlkTanom[combined_idx][is_early], 
    c='blue',
    s=60, alpha=0.8, edgecolors='none',
    label='1980-1999'
)
ax.scatter(
    BlkPanom[combined_idx][is_late], BlkTanom[combined_idx][is_late], 
    c='red',
    s=60, alpha=0.8, edgecolors='none',
    label='2000-2021'
)
ax.legend(loc='upper right')

# add x=0 and y=0 lines
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
# axises
ax.set_ylim(-2.5, 2.5)
ax.set_xlim(-vabs*1.1, vabs*1.1)
ax.set_xlabel('Precipitation Anomaly')
ax.set_ylabel('Temperature Anomaly')
ax.set_title('Blocking Events: T vs P Anomaly\nColored by periods')

plt.tight_layout()
plt.show()

plt.savefig('SchematicPlotPanela_TwithP_anom_colortime_Top100.png', dpi=300)
plt.close()
print('fig1 done')


# fig2 -------------------------
# classify in periods

fig, ax = plt.subplots(figsize=(7, 6))

norm = plt.Normalize(vmin=np.nanmin(BlkIntensity[combined_idx]), vmax=np.nanmax(BlkIntensity[combined_idx]))

sc1 = ax.scatter(
    BlkPanom[combined_idx][is_early], BlkTanom[combined_idx][is_early], 
    c=BlkIntensity[combined_idx][is_early],marker='o',
    s=60, alpha=0.8, edgecolors='none',cmap='hot_r',norm=norm,
    label='1980-1999'
)
ax.scatter(
    BlkPanom[combined_idx][is_late], BlkTanom[combined_idx][is_late], 
    c=BlkIntensity[combined_idx][is_late],marker='^',
    s=60, alpha=0.8, edgecolors='none',cmap='hot_r',norm=norm,
    label='2000-2021'
)

ax.legend(loc='upper right')
cbar = plt.colorbar(sc1, ax=ax)
cbar.set_label('Blocking Intensity')

# add x=0 and y=0 lines
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
# axises
pvabs = np.nanmax(np.abs(BlkPanom[combined_idx]))
ax.set_ylim(-2.5, 2.5)
ax.set_xlim(-pvabs*1.1, pvabs*1.1)
ax.set_xlabel('Precipitation Anomaly')
ax.set_ylabel('Temperature Anomaly')
ax.set_title('Blocking Events: T vs P Anomaly\nColored by intensity')

plt.tight_layout()
plt.show()

plt.savefig('SchematicPlotPanela_TwithP_anom_colorIntensity_Top100.png', dpi=300)
plt.close()
print('fig2 done')