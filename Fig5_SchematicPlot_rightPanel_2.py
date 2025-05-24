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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.path as mpath
from matplotlib.lines import Line2D

# 1. blocking labels
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily", "rb") as fp:
    Blocking_diversity_date = pickle.load(fp)      
Blocking_allType_date = [item for sublist in Blocking_diversity_date for item in sublist]
print('blocking data read in', flush=True)

# # # 2. temperature (T) anomaly (3d: time, lat, lon) - 0.25drgee!
# lats_TS = np.arange(20, 70.01, 0.25)  
# print(lats_TS,flush=True) # 20-70, by 0.25
# lons_TS = np.arange(0, 360, 0.25)
# print(lons_TS,flush=True) # 0-359.75, by 0.25
# Tanomly = np.load('/scratch/bell/hu1029/Data/processed/ERA5_Tsurface_1980_2021_20_70N_dailyAnomaly.npy')
# print(Tanomly.shape, flush=True) # 15342, 201, 1440
# print(len(lats_TS), flush=True) # 
# print(len(lons_TS), flush=True) # 
# # time for Tanomaly
# Datestamp_TS = pd.date_range(start="1980-01-01", end="2021-12-31")
# Date0_TS = pd.DataFrame({'date': pd.to_datetime(Datestamp_TS)})
# timestamp_TS = list(Date0_TS['date'])
# print(len(timestamp_TS),flush=True)

# # # 3. TCWV anomaly (3d: time, lat, lon)
# TCWVanomly = np.load('/scratch/bell/hu1029/Data/processed/ERA5_Prep_1980_2021_20_70N_dailyAnomaly_5dmovingMean.npy')
lats_P = np.arange(20, 71, 1)  
print(lats_P,flush=True) # 20-70, by 1
lons_P = np.arange(0, 360, 1)
print(lons_P,flush=True) 
# print(TCWVanomly.shape, flush=True) # 15342, 51, 360
# print(len(lats_P), flush=True) #
# print(len(lons_P), flush=True) #

# # # # 4. geopotential height anomaly (3d: time, lat, lon; event-based, selected from blocking index)
# # # # get the Z500anom, 1dg (same as LWA)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
Zanom_daily = ds['z'].squeeze().resample(time='1D').mean()
Zanom = np.array(Zanom_daily)
print(np.nanmean(Zanom), flush=True)
Zlon = np.array(ds['lon'])
Zlat = np.array(ds['lat'])
print(Zlat, flush=True)
print(Zlon, flush=True)
midlat = np.where((Zlat >= 20) & (Zlat <= 70))[0]
Zlat = Zlat[midlat]
print(Zlat, flush=True)
# Zanom = Zanom[:,midlat,:] # increasing from 0-90N
# print(Zanom.shape, flush=True)
# print('-------- Zanom loaded --------', flush=True)
# # time for Z500
# Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
# Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
# timestamp = list(Date0['date'])
# timestamparr = np.array(timestamp)
# print(len(timestamp), flush=True)

# load 
data = np.load('/scratch/bell/hu1029/blocking_event_data_precpRaw5dmovingMean_Intensity.npz', allow_pickle=True)
BlkTanom = data['BlkTanom']
BlkPanom = data['BlkPanom']
BlkDuration = data['BlkDuration']
BlkTypeid = data['BlkTypeid']
BlkpeakLonlist = data['BlkpeakLonlist']
BlkTimeid = data['BlkTimeid']
BlkIntensity = data['BlkIntensity']
targetEeventid = data['targetEeventid']

# four event indices
BlkTimeid_dt = pd.to_datetime(BlkTimeid)
is_JJA = BlkTimeid_dt.month.isin([6, 7, 8])
is_DJF = BlkTimeid_dt.month.isin([12, 1, 2])
JJA_top100_idx = np.argsort(BlkIntensity[is_JJA])[-100:]  
DJF_top100_idx = np.argsort(BlkIntensity[is_DJF])[-100:]  
JJA_global_idx = np.where(is_JJA)[0][JJA_top100_idx]
DJF_global_idx = np.where(is_DJF)[0][DJF_top100_idx]
combined_idx = np.concatenate([JJA_global_idx, DJF_global_idx]) # is the idx in BlkTimeid_dt
# targetEeventid is the real event id in all blocking events

# T_plot_list = []
# P_plot_list = []
# Z_plot_list = []
# lons_TS_plot_list = []
# lons_P_plot_list = []
# Zlon_plot_list = []

# for i in combined_idx:

#     ev = targetEeventid[i] # the real location in the all events
#     print('event in BlkTimeid_dt location:', i, flush=True)
#     print('event in totalevent location:', ev, flush=True)
#     dayi = Blocking_allType_date[ev]
#     print(dayi, flush=True)
#     timei_inTS = [timestamp_TS.index(val) for val in dayi] # get the time index
#     timei_inZ = [timestamp.index(val) for val in dayi] # get the time index
#     eventTanom = Tanomly[timei_inTS, :, :]
#     eventTCWVanom = TCWVanomly[timei_inTS, :, :]
#     eventZanom = Zanom[timei_inZ, :, :]

#     eventTanom_avg = np.nanmean(eventTanom, axis=0)
#     eventPanom_avg = np.nanmean(eventTCWVanom, axis=0)
#     eventZanom_avg = np.nanmean(eventZanom, axis=0)

#     T_plot_list.append(eventTanom_avg)
#     P_plot_list.append(eventPanom_avg)
#     Z_plot_list.append(eventZanom_avg)

#     print(f'combined_idx {i}, -------------', flush=True)

# # save the list
# # the list for all the selected events (200 in total)
# with open('/scratch/bell/hu1029/Data/processed/T_plot_list_top100_2', 'wb') as f:
#     pickle.dump(T_plot_list, f)
# with open('/scratch/bell/hu1029/Data/processed/P_plot_list_top100_2', 'wb') as f:
#     pickle.dump(P_plot_list, f)
# with open('/scratch/bell/hu1029/Data/processed/Z_plot_list_top100_2', 'wb') as f:
#     pickle.dump(Z_plot_list, f)

# now reload the list
with open('/scratch/bell/hu1029/Data/processed/T_plot_list_top100_2', 'rb') as f:
    T_plot_list = pickle.load(f)
with open('/scratch/bell/hu1029/Data/processed/P_plot_list_top100_2', 'rb') as f:
    P_plot_list = pickle.load(f)
with open('/scratch/bell/hu1029/Data/processed/Z_plot_list_top100_2', 'rb') as f:
    Z_plot_list = pickle.load(f)

lats_TS = np.arange(20, 70.01, 0.25)  
print(lats_TS,flush=True) # 20-70, by 0.25
lons_TS = np.arange(0, 360, 0.25)

lats_P = np.arange(20, 71, 1)  
print(lats_P,flush=True) # 20-70, by 1
lons_P = np.arange(0, 360, 1)
print(lons_P,flush=True) 

# make the countour plot
BlkTimeid_dt = pd.to_datetime(BlkTimeid)
is_JJA = BlkTimeid_dt.month.isin([6, 7, 8])
is_DJF = BlkTimeid_dt.month.isin([12, 1, 2])
JJA_top100_idx = np.argsort(BlkIntensity[is_JJA])[-100:]  
DJF_top100_idx = np.argsort(BlkIntensity[is_DJF])[-100:]  
JJA_global_idx = np.where(is_JJA)[0][JJA_top100_idx]
DJF_global_idx = np.where(is_DJF)[0][DJF_top100_idx]
combined_idx = np.concatenate([JJA_global_idx, DJF_global_idx]) # is the idx in BlkTimeid_dt


fig = plt.figure(figsize=(10, 16), constrained_layout=True)
axes = []
event_names = ['ev1', 'ev2', 'ev3', 'ev4']
# the index of events in the BlkTimeid_dt
ev1 = 576 #676
ev2 = 1523
ev3 = 84
ev4 = 1359
evlist = [ev1, ev2, ev3, ev4]
P_plot_lim = 0.002
lonrange = 55

for i in range(4):

    evid = evlist[i]
    print(evid, flush=True)
    nindex = np.where(combined_idx == evid)[0][0]
    print(nindex, flush=True)
    T_plot = T_plot_list[nindex]
    P_plot = P_plot_list[nindex]
    Z_plot = Z_plot_list[nindex]
    Eventdates = Blocking_allType_date[targetEeventid[evid]]
    print(Eventdates, flush=True)
    print('Peaking date:')
    print(BlkTimeid[evid], flush=True)
    firstday = Eventdates[0]
    lastday = Eventdates[-1]
    center_lon = BlkpeakLonlist[evid]
    print(center_lon, flush=True)

    proj = ccrs.PlateCarree(central_longitude=center_lon)
    ax = fig.add_subplot(4, 1, i+1, projection=proj)
    ax.set_extent([center_lon - lonrange, center_lon + lonrange, 20, 70], crs=ccrs.PlateCarree())
    ax.set_aspect('auto')
    axes.append(ax)
    # plot
    lon_min = center_lon - lonrange
    lon_max = center_lon + lonrange
    lat_min = 20
    lat_max = 70
    wrapped_lon_min = lon_min % 360
    wrapped_lon_max = lon_max % 360
    if wrapped_lon_min < wrapped_lon_max:
        lon_mask = (lons_TS >= wrapped_lon_min) & (lons_TS <= wrapped_lon_max)
    else:
        lon_mask = (lons_TS >= wrapped_lon_min) | (lons_TS <= wrapped_lon_max)
    lat_mask = (lats_TS >= lat_min) & (lats_TS <= lat_max)
    T_plot_visible = T_plot[np.ix_(lat_mask, lon_mask)]
    vabsmax = np.nanmax(np.abs(T_plot_visible))
    norm = matplotlib.colors.Normalize(vmin=-vabsmax, vmax=vabsmax)

    # vabs = np.abs(T_plot)
    # vabsmax = np.nanmax(vabs)
    # norm = matplotlib.colors.Normalize(vmin=-vabsmax, vmax=vabsmax)

    ax.add_feature(cfeature.COASTLINE.with_scale('110m'),color='black', linewidth=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray',linestyle='--',alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    cs1 = ax.contourf(lons_TS, lats_TS, T_plot, levels=10, cmap='RdBu_r', norm=norm,
                        alpha=0.9, transform=ccrs.PlateCarree(),extend='both')
    cbar = fig.colorbar(cs1, ax=ax, orientation='vertical', pad=0.05, extend='both')
    cbar.set_label("T Anomaly")
    
    # 2. P_plot > 0.01
    pmax = np.nanmax(P_plot)
    print(pmax, flush=True)
    if pmax > P_plot_lim:
        mask_pos = np.where(P_plot > P_plot_lim, P_plot, np.nan)
        cs_pos = ax.contourf(
            lons_P, lats_P, mask_pos,
            levels=[P_plot_lim, np.nanmax(P_plot)], 
            hatches=['...'], colors='none', 
            transform=ccrs.PlateCarree()
    )
    # 3. P_plot < -0.01 
    pmin = np.nanmin(P_plot)
    print(pmin, flush=True)
    if pmin < -P_plot_lim:
        mask_neg = np.where(P_plot < -P_plot_lim, P_plot, np.nan)
        cs_neg = ax.contourf(
            lons_P, lats_P, mask_neg,
            levels=[np.nanmin(P_plot), -P_plot_lim],
            hatches=['XXX'], colors='none',
            transform=ccrs.PlateCarree()
        )

    cs3 = ax.contour(Zlon, Zlat, Z_plot, levels=14, colors='blue', linewidths=1, 
                        transform=ccrs.PlateCarree())
    cs3.clabel(cs3.levels[::2], inline=True, fontsize=8, fmt='%.1f')

    ax.set_title(f"Blocking event {i+1}: {firstday.strftime('%Y-%m-%d')} to {lastday.strftime('%Y-%m-%d')}", fontsize=12)
    ax.set_ylabel("Latitude")

# cbar = fig.colorbar(ScalarMappable(cmap='RdBu_r', norm=norm),
#     ax=axes,orientation='horizontal',pad=0.05)
# cbar.set_label("T Anomaly")

plt.show()
plt.savefig('SchematicPlotRightpanel.png', dpi=300)

print('fig done', flush=True)


# separate the figure

for i in range(4):
    evid = evlist[i]
    nindex = np.where(combined_idx == evid)[0][0]
    T_plot = T_plot_list[nindex]
    P_plot = P_plot_list[nindex]
    Z_plot = Z_plot_list[nindex]
    Eventdates = Blocking_allType_date[targetEeventid[evid]]
    firstday = Eventdates[0]
    lastday = Eventdates[-1]
    center_lon = BlkpeakLonlist[evid]

    fig = plt.figure(figsize=(8, 4), constrained_layout=True)
    proj = ccrs.PlateCarree(central_longitude=center_lon)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([center_lon - lonrange, center_lon + lonrange, 20, 70], crs=ccrs.PlateCarree())
    ax.set_aspect('auto')

    vabs = np.abs(T_plot)
    vabsmax = np.nanmax(vabs)
    norm = matplotlib.colors.Normalize(vmin=-vabsmax, vmax=vabsmax)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'),color='black', linewidth=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray',linestyle='--',alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    cs1 = ax.contourf(lons_TS, lats_TS, T_plot, levels=10, cmap='RdBu_r', norm=norm,
                         transform=ccrs.PlateCarree())
    
    # 2. P_plot > 0.01
    pmax = np.nanmax(P_plot)
    print(pmax, flush=True)
    if pmax > P_plot_lim:
        mask_pos = np.where(P_plot > P_plot_lim, P_plot, np.nan)
        cs_pos = ax.contourf(
            lons_P, lats_P, mask_pos,
            levels=[P_plot_lim, np.nanmax(P_plot)], 
            hatches=['...'], colors='none',  
            transform=ccrs.PlateCarree()
        )
    # 3. P_plot < -0.01
    pmin = np.nanmin(P_plot)
    print(pmin, flush=True)
    if pmin < -P_plot_lim:
        mask_neg = np.where(P_plot < -P_plot_lim, P_plot, np.nan)
        cs_neg = ax.contourf(
            lons_P, lats_P, mask_neg,
            levels=[np.nanmin(P_plot), -P_plot_lim],
            hatches=['XXX'], colors='none',
            transform=ccrs.PlateCarree()
        )

    cs3 = ax.contour(Zlon, Zlat, Z_plot, levels=14, colors='blue', linewidths=1, 
                        transform=ccrs.PlateCarree())
    cs3.clabel(cs3.levels[::2], inline=True, fontsize=8, fmt='%.1f')

    ax.set_title(f"Blocking event {i+1}: {firstday.strftime('%Y-%m-%d')} to {lastday.strftime('%Y-%m-%d')}", fontsize=12)

    cbar = fig.colorbar(cs1, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label("T Anomaly")

    fname = f"SchematicPlotRightPanel_event{i+1}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"{fname} saved.", flush=True)

    