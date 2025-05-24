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
import matplotlib.colors as mcolors
from matplotlib import cm

def findClosest(lati, latids):

    if isinstance(lati, np.ndarray):  # if lat is an array
        closest_indices = []
        for l in lati:  
            diff = np.abs(l - latids)
            closest_idx = np.argmin(diff) 
            closest_indices.append(closest_idx)
        return closest_indices
    else:
        # if lat is a single value
        diff = np.abs(lati - latids)
        return np.argmin(diff) 

# read the fire data --------------------
df = pd.read_csv("/home/hu1029/FRAR/Fire/fire_nrt_M-C61_595038.csv")
# crop the target region
df_targetregion = df[(df['longitude'] >= -130) & (df['latitude'] >= 30) & (df['longitude'] <= -100) & (df['latitude'] <= 50)].copy()
df_targetregion['acq_date'] = pd.to_datetime(df_targetregion['acq_date'])
df_jan2025 = df_targetregion[(df_targetregion['acq_date'].dt.year  == 2025) & (df_targetregion['acq_date'].dt.month == 1)].copy()
# convert latitude, longitude values into point geometry
gdf = geopandas.GeoDataFrame(
    df_jan2025, geometry=geopandas.points_from_xy(df_jan2025.longitude, df_jan2025.latitude), crs="EPSG:4326"
)

# read in Z500 data ------------------
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_geopotential500_global_20250108to10.nc')
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
# select a time
zs_range = z500_sub.sel(
    valid_time=slice('2025-01-08T00:00', '2025-01-09T00:00')
)
zs_mean = zs_range.mean(dim='valid_time')
zs = zs_mean.squeeze()  # remove the redundant dimension
print(np.nanmax(zs))
print(np.nanmin(zs))

# read the wind data --------------------
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_UV_20250108_hourly.nc')
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

# read in T2m data ----------------
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_2mT_20250108_hourly.nc')
t2m = ds['t2m'] - 273.15
t2m_sub = t2m.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
t2m_mean = t2m_sub.mean(dim='valid_time')
t2m_mean = t2m_mean.squeeze()  # remove the redundant dimension
t2m_mean = np.array(t2m_mean)
print(np.nanmax(t2m_mean))
print(np.nanmin(t2m_mean))

# read in dT2m data ----------------
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_dT2m_20250108_hourly.nc')
dT2m = ds['d2m'] - 273.15
dT2m_sub = dT2m.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
dT2m_mean = dT2m_sub.mean(dim='valid_time')
dT2m_mean = dT2m_mean.squeeze()  # remove the redundant dimension
dT2m_mean = np.array(dT2m_mean)

# calculate the VPD ------------------
def es(Tc):

    Tc = np.asarray(Tc)
    A = np.where(Tc < 0, 21.87, 17.27)
    B = np.where(Tc < 0, 265.5, 237.15)

    return 0.611 * np.exp(A * Tc / (Tc + B))

e_s  = es(t2m_mean)   # (kPa)
e_a  = es(dT2m_mean)  # (kPa)
VPD = e_s - e_a 
print(np.nanmax(VPD))
print(np.nanmin(VPD))

# calculate the climatology VPD ---------------
# t2m
ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_t2m_1981_2020_0108_hourly.nc')
t2m = ds['t2m'] - 273.15
t2m_sub = t2m.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
t2m_sub = t2m_sub.squeeze()  # remove the redundant dimension
t2m_sub = np.array(t2m_sub)

ds = xr.open_dataset('/home/hu1029/FRAR/ERA5_d2m_1981_2020_0108_hourly.nc')
dT2m = ds['d2m'] - 273.15
dT2m_sub = dT2m.sel(
    latitude=slice(latmin, latmax),
    longitude=slice(lonmin, lonmax)
)
dT2m_sub = dT2m_sub.squeeze()  # remove the redundant dimension
dT2m_sub = np.array(dT2m_sub)

e_s_19812020 = es(t2m_sub)   # (kPa)
e_a_19812020 = es(dT2m_sub)  # (kPa)
VPD_19812020 = e_s_19812020 - e_a_19812020
VPD_clima = np.nanmean(VPD_19812020, axis=0)

VPD_anomaly = VPD - VPD_clima
print(np.nanmax(VPD_anomaly))
print(np.nanmin(VPD_anomaly))

# make the plot -----------------------------
gdf_sorted = gdf.sort_values('frp')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m', linewidth=0.8, color = 'darkgray')
# ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
# ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='darkgray')
ax.add_feature(cfeature.STATES, edgecolor='darkgray', linewidth=0.5)

ax.set_extent([-180, -100, 10, 70], crs=ccrs.PlateCarree())

# make the color class
bounds = [0, 25, 50, 75, 100]
cmap   = ListedColormap(['#ff9933','#ff6600','#ff3300','#cc0000'])
cmap.set_over('#800000')  
norm   = BoundaryNorm(bounds, ncolors=cmap.N)

# 01 VPD anomaly
# make the middle color white
levels = np.arange(-0.7, 1.3, 0.2)
cmap_vpd = cm.get_cmap('PiYG_r', 18)
colors = cmap_vpd(np.linspace(0, 1, 18))
selectColors = colors[[0,2,4,6,9, 12,13,14,15,16,17]]
new_cmap = ListedColormap(selectColors)
norm_vpd = BoundaryNorm(boundaries=levels, ncolors=len(selectColors), extend='both')

cs0 = ax.contourf(zlon_sub, zlat_sub, VPD_anomaly, extend='both',levels=levels,norm=norm_vpd, cmap=new_cmap) #PuRd
cbar = plt.colorbar(cs0, ax=ax, orientation='horizontal', extend = 'max', pad=0.08, aspect=30)
cbar.set_label('VPD Anomaly (KPa)')

# 03 wind barbs
barbs = ax.barbs(
    lon2d_s, lat2d_s,
    u_s, v_s,
    length=6,
    pivot='middle',
    color='gray',
    alpha = 0.8,
    transform=ccrs.PlateCarree()
)

# 04 countours
levels = np.arange(5000, 6000, 50)
cs = ax.contour(zlon_sub, zlat_sub, zs.values, colors='black',
                    levels=levels,transform=ccrs.PlateCarree(),linewidths=1)
ax.clabel(cs, inline=True, fontsize=10, fmt="%1.0f",use_clabeltext=True, manual=False, levels=cs.levels[::2])


# 02 firepoint
sc = ax.scatter(gdf_sorted.geometry.x, gdf_sorted.geometry.y, c=gdf_sorted['frp'], 
                    cmap=cmap, norm=norm, s=10, marker='^', alpha=0.7 ,transform=ccrs.PlateCarree())
cbar2 = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.08, aspect=30, extend='max')
cbar2.set_label('Fire Radiative Power (MW)')

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

ax.set_title(f'Contours: Z500 at \n 2025-01-08 UTC')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.show()
plt.savefig(f'/home/hu1029/FRAR/DipoleFire_FRPscatter_20250108_withVPDanomaly.png')

print('done')

