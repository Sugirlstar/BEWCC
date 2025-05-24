import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
import math
import time
import geopandas

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
from multiprocessing import Pool, Manager
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
import imageio
from scipy import stats
from collections import defaultdict
from scipy.ndimage import label
from scipy.interpolate import interp2d
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime, timedelta

# read in Z500 data ------------------
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_geopotential500_global_202502_daily00.nc')
z500 = ds['z'] / 9.81
# range
latmin, latmax = 70, 10
lonmin, lonmax = 180, 260  # lon is 0-360
# crop the z500
z500_sub = z500.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
# get the lat and lon values
zlat_sub = z500_sub.latitude.values  # shape (nlat,)
zlon_sub = z500_sub.longitude.values # shape (nlon,)

timepoint = '2025-02-02T00:00'
# timepoint = date_dt[i]
zs = z500_sub.sel(valid_time=timepoint) 
zs = zs.squeeze()  # remove the redundant dimension

# read the IWV data --------------------
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_columnwatervapor_20250202_hourly.nc')
tcwv = ds['tcwv']
# crop the tcwv
tcwv_sub = tcwv.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
tcwv_mean = tcwv_sub.mean(dim='valid_time')
tcwv_mean = tcwv_mean.squeeze()  # remove the redundant dimension
print(np.nanmax(tcwv_mean))
print(np.nanmin(tcwv_mean))
tcwv_mean = np.array(tcwv_mean)

# read in precipitation data --------------------
ds_cloud = xr.open_dataset(f"/home/hu1029/FRAR/AR/IMERG_0202.nc4")
prep = ds_cloud['precipitation']
preplat = ds_cloud['lat']
preplon = ds_cloud['lon']
prep_sub = prep.sel(
    lat=slice(10, 70),
    lon=slice(-180,-100)
)
preplat_sub = prep_sub.lat.values  
preplon_sub = prep_sub.lon.values 
prep_day = prep_sub.sel(time=timepoint)
prep_day = prep_day.squeeze()  # remove the redundant dimension
prep_day = np.array(prep_day).T

# read the wind data --------------------
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_UV_20250202_hourly.nc')
u10 = ds['u']
u10_sub = u10.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
u10mean = u10_sub.mean(dim='valid_time')
u10mean = u10mean.squeeze()  # remove the redundant dimension
v10 = ds['v']
v10_sub = v10.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
v10mean = v10_sub.mean(dim='valid_time')
v10mean = v10mean.squeeze()  # remove the redundant dimension

# get the value each 5 grids
lon2d, lat2d = np.meshgrid(zlon_sub, zlat_sub)
step = 20
u_s = u10mean.values[::step, ::step]
v_s = v10mean.values[::step, ::step]
lon2d_s = lon2d[::step, ::step]
lat2d_s = lat2d[::step, ::step]


# make the plot  -----------------
levels = np.arange(5000, 6000, 50)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m', linewidth=0.8, color = 'darkgray')
ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='darkgray')
ax.set_extent([-180, -100, 10, 70], crs=ccrs.PlateCarree())

cs1 = ax.contourf(zlon_sub, zlat_sub, tcwv_mean, cmap='YlGnBu')
cbar = plt.colorbar(cs1, ax=ax, orientation='vertical', pad=0.08, aspect=30, extend='max')
cbar.set_label('TCWV (kg/m2)', fontsize=12)

# wind barbs
barbs = ax.barbs(
    lon2d_s, lat2d_s,
    u_s, v_s,
    length=6,
    pivot='middle',
    color='gray',
    alpha = 0.8,
    transform=ccrs.PlateCarree()
)

cs = ax.contour(zlon_sub, zlat_sub, zs.values, colors='black',
                    levels=levels,transform=ccrs.PlateCarree(),linewidths=1)
ax.clabel(cs, inline=True, fontsize=10, fmt='%1.0f')

maxv = np.nanmax(prep_day)
cs_hatch = ax.contourf(preplon_sub, preplat_sub, prep_day, levels=[10, maxv],
        colors='none',hatches=['///'], transform=ccrs.PlateCarree())

# ticks
xticks = np.arange(-180, -100+1, 20)   # e.g. [180,190,…,240]
yticks = np.arange(10, 70+1, 20)   # e.g. [20,30,…,80]
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(which='major',
            top=True, bottom=True,
            left=True, right=True,
            labeltop=True, labelbottom=True,
            labelleft=True, labelright=True)
# gdf.plot(ax=ax, color="red", markersize=1)

ax.set_title(f'Contours: Z500 at \n{timepoint} UTC')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.show()
plt.savefig(f'/home/hu1029/FRAR/DipoleAR_TCWV_PrepHatches_{timepoint}_withWind.png')

print('done')
