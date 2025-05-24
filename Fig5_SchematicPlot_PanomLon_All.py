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

extendlon = 179
lowp = 30
highp = 70
lats_P = np.arange(20, 71, 1)  
lons_P = np.arange(0, 360, 1)

# load 
data = np.load('/scratch/bell/hu1029/blocking_event_data_precp5dmovingMean_Intensity.npz', allow_pickle=True)
BlkTanom = data['BlkTanom']
BlkPanom = data['BlkPanom']
BlkDuration = data['BlkDuration']
BlkTypeid = data['BlkTypeid']
BlkpeakLonlist = data['BlkpeakLonlist']
BlkTimeid = data['BlkTimeid']
BlkIntensity = data['BlkIntensity']

# event indices
combined_idx = np.arange(len(BlkTimeid)) 
# targetEeventid is the real event id in all blocking events

# now reload the list
with open('/scratch/bell/hu1029/Data/processed/T_plot_list_allEvents', 'rb') as f:
    T_plot_list = pickle.load(f)
with open('/scratch/bell/hu1029/Data/processed/P_plot_list_allEvents', 'rb') as f:
    P_plot_list = pickle.load(f)
with open('/scratch/bell/hu1029/Data/processed/Z_plot_list_allEvents', 'rb') as f:
    Z_plot_list = pickle.load(f)

# composite dry / wet events
# get the lon center of all dry/wet block events
wetthreshold = 0.0001
T200_centerlons = BlkpeakLonlist[combined_idx]
T200_Panom = BlkPanom[combined_idx]
wetindex = np.where(T200_Panom > wetthreshold)[0]
dryindex = np.where(T200_Panom < -wetthreshold)[0]
print('wet index:', flush=True)
print(wetindex, flush=True)
print('dry index:', flush=True)
print(dryindex, flush=True)
Pevents_dry = np.array([P_plot_list[i] for i in dryindex])
Pevents_wet = np.array([P_plot_list[i] for i in wetindex])
print(Pevents_dry.shape) # [events,lat,lon]
print(Pevents_wet.shape) # [events,lat,lon]
Pevents_Lonmean_dry = np.nanmean(Pevents_dry, axis=1) 
Pevents_Lonmean_wet = np.nanmean(Pevents_wet, axis=1)
print(Pevents_Lonmean_dry.shape)
print(Pevents_Lonmean_wet.shape)
# get the slice centered at the center lon
drycenter = [T200_centerlons[i] for i in dryindex]
wetcenter = [T200_centerlons[i] for i in wetindex]
print(drycenter)
print(wetcenter)
Pevents_Lonmean_dry_list = []
for i,dryindex in enumerate(drycenter):
    center_idx = np.where(lons_P == dryindex)[0][0] 
    idx_range = (center_idx + np.arange(-extendlon, extendlon+1)) % len(lons_P)
    dryslice = Pevents_Lonmean_dry[i,idx_range]
    Pevents_Lonmean_dry_list.append(dryslice)
Pevents_Lonmean_dry_arr = np.array(Pevents_Lonmean_dry_list)
Pevents_Dry = np.nanmean(Pevents_Lonmean_dry_arr, axis=0)
Pevents_Dry_lowrange = np.percentile(Pevents_Lonmean_dry_arr, lowp, axis=0)
Pevents_Dry_highrange = np.percentile(Pevents_Lonmean_dry_arr, highp, axis=0)
# Pevents_Dry_lowrange = Pevents_Dry - np.nanstd(Pevents_Lonmean_dry_arr, axis=0)
# Pevents_Dry_highrange = Pevents_Dry + np.nanstd(Pevents_Lonmean_dry_arr, axis=0)
print(Pevents_Dry)
print(np.nanstd(Pevents_Lonmean_dry_arr, axis=0))
Pevents_Lonmean_wet_list = []
for i,wetindex in enumerate(wetcenter):
    center_idx = np.where(lons_P == wetindex)[0][0] 
    idx_range = (center_idx + np.arange(-extendlon, extendlon+1)) % len(lons_P)
    wetslice = Pevents_Lonmean_wet[i,idx_range]
    Pevents_Lonmean_wet_list.append(wetslice)
Pevents_Lonmean_wet_arr = np.array(Pevents_Lonmean_wet_list)
Pevents_Wet = np.nanmean(Pevents_Lonmean_wet_arr, axis=0)
Pevents_Wet_lowrange = np.percentile(Pevents_Lonmean_wet_arr, lowp, axis=0)
Pevents_Wet_highrange = np.percentile(Pevents_Lonmean_wet_arr, highp, axis=0)
# Pevents_Wet_lowrange = Pevents_Wet - np.nanstd(Pevents_Lonmean_wet_arr, axis=0)
# Pevents_Wet_highrange = Pevents_Wet + np.nanstd(Pevents_Lonmean_wet_arr, axis=0)
print(Pevents_Wet.shape)

# make the plot ------------------
x = np.arange(-extendlon, extendlon+1)
cmap = plt.get_cmap('BrBG')
brown = cmap(0.2)
green = cmap(0.8)

# make the plot
plt.figure(figsize=(8, 4))
plt.plot(x, Pevents_Dry, color=brown, label='dry')
plt.plot(x, Pevents_Wet, color=green, label='wet')
# add x=0
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.ylim(-0.0006, 0.0006)
# add the legend
plt.xlabel('Longitude relative to center (°)')
plt.ylabel('Panom average')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(f'Panom_Lon_All_extend_{extendlon}dg.png' , dpi=300)
plt.close()

# make the plot with range shading -------------------
x = np.arange(-extendlon, extendlon + 1)
cmap = plt.get_cmap('BrBG')
brown = cmap(0.2)
green = cmap(0.8)

Pevents_Dry = Pevents_Dry * 1000
Pevents_Wet = Pevents_Wet * 1000
Pevents_Dry_lowrange = Pevents_Dry_lowrange * 1000
Pevents_Dry_highrange = Pevents_Dry_highrange * 1000
Pevents_Wet_lowrange = Pevents_Wet_lowrange * 1000
Pevents_Wet_highrange = Pevents_Wet_highrange * 1000
 
plt.figure(figsize=(8, 4))

plt.plot(x, Pevents_Dry, color=brown, label='Dry blocks', linewidth=2)
plt.fill_between(x, Pevents_Dry_lowrange, Pevents_Dry_highrange, 
                 color=brown, alpha=0.3, linewidth=0)
plt.plot(x, Pevents_Wet, color=green, label='Wet blocks', linewidth=2)
plt.fill_between(x, Pevents_Wet_lowrange, Pevents_Wet_highrange, 
                 color=green, alpha=0.3, linewidth=0)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

plt.ylim(-0.9, 0.9)
plt.xlabel('Longitude relative to center (°)')
plt.ylabel('Panom (mm/day)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f'Panom_Lon_All_extend_{extendlon}dg_range{lowp}to{highp}.png', dpi=300)
plt.show()
plt.close()
