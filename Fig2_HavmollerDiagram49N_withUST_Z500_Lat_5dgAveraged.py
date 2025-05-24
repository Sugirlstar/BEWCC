import numpy as np
from multiprocessing import Pool, Manager
import math
import cv2
import os
from itertools import combinations
from netCDF4 import Dataset
from HYJfunction import *
from datetime import date
import matplotlib.pyplot as plt
import regionmask
import datetime as dt
import pandas as pd
import xarray as xr
import matplotlib.dates as mdates


# # function
# def anomalies_seasons(df_VAR):
#     import numpy as np
#     import pandas as pd
#     import datetime as dt
#     ANOMA = df_VAR * np.nan

#     def custom_strptime(time_data):
#         if len(time_data) == 7:
#             month = time_data[:2]
#             day = time_data[2:4]
#             year = '0' + time_data[4:]
#             date_str = f'{month}{day}{year}'
#             return dt.datetime.strptime(date_str, '%m%d%Y')
#         else:
#             return dt.datetime.strptime(time_data, '%m%d%Y')
    
#     # dates_d = np.array([dt.datetime.strptime(iii, '%m%d%Y') for iii in df_VAR.index])
#     dates_d = np.array([custom_strptime(iii) for iii in df_VAR.index])
#     for i in np.arange(1, 13):
#         mes = df_VAR.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]] 
#         dates_d_mes = dates_d[np.where(np.array([ii.month for ii in dates_d])==i)[0]] 
#         temp = mes * np.nan
#         Nodays = np.array([ii.day for ii in dates_d_mes]).max()
#         if np.isnan(Nodays) == True: continue
#         for j in np.arange(1, Nodays + 1):
#             dia = mes.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]]
#             media = dia.mean() 
#             anoma = dia - media
#             temp.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]] = anoma
#         ANOMA.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]] = temp
#     return ANOMA

# # read in LWA
# ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
# Zanom_daily = ds['z'].squeeze().resample(time='1D').mean()
# Zanom = np.array(Zanom_daily)
# lon = np.array(ds['lon'])
# lat = np.array(ds['lat'])
# print(lat, flush=True)
# print(lon, flush=True)
# print(Zanom.shape, flush=True)
# print('-------- Zanom loaded --------', flush=True)
# # get the region
# Ls = 49
# Lsindex = np.where((lat >= Ls-2) & (lat <= Ls+2))[0]
# LWA_td = Zanom[:, Lsindex, :]
# LWA_td_test = Zanom[:, np.where(lat==Ls)[0], :]
# print('see if there are differences, original:')
# print(LWA_td_test.shape)
# LWA_td = np.nanmean(LWA_td, axis=1)
# print('after mean axis=1:')
# print(LWA_td.shape)
# lon_eventmin = 150; lon_eventmax = 270
# lonst = np.where((lon >= lon_eventmin) & (lon <= lon_eventmax))[0]
# LWA_td = LWA_td[:, lonst]
# lwafinallon = lon[lonst]
# print(LWA_td.shape)

#     # get the target time
# t = np.arange(date(1979,1,1).toordinal(),date(2021,12,31).toordinal()+1) # the target time
# stday = date(2021,6,10).toordinal()
# edday = date(2021,7,5).toordinal()
# stindex = np.where(t == stday)[0][0]
# edindex = np.where(t == edday)[0][0]
# LWAevent = LWA_td[stindex:edindex+1,:]
# print(LWAevent.shape)

# # read in the data
# tnp = np.load('/scratch/bell/hu1029/Karens/SST_OISST_cropped.npy')
# # tnp = np.load('/scratch/bell/hu1029/Karens/SST_OISST_cropped.npy')
# # OceanMask = np.where(~np.isnan(tnp[0,:,:]), 1, np.nan)
# print(tnp.shape)

# ncfile = xr.open_dataset('/scratch/bell/hu1029/Data/processed/TS_ERA5.nc')
# t_k = np.array(ncfile['TS'])
# lats = np.array(ncfile['lat'])
# lons = np.array(ncfile['lon'])
# ncfile.close()
# lat_minHW = 0; lat_maxHW = 72; lon_minHW = 150; lon_maxHW = 270
# pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))[0]
# pos_lons = np.where((lons >= lon_minHW) & (lons <= lon_maxHW))[0]
# ts_lats = lats[pos_lats]
# ts_lons = lons[pos_lons]
# ts = t_k-273.15
# ts = ts[:, pos_lats, :]
# ts = ts[:, :, pos_lons]

# # Lsindex = np.where(ts_lats == Ls)[0][0]
# Lsindex = np.where((ts_lats >= Ls-2) & (ts_lats <= Ls+2))[0]
# ts = ts[:, Lsindex, :]
# ts = np.nanmean(ts, axis=1)
# lon_eventmin = 150; lon_eventmax = 270
# lonst = np.where((ts_lons >= lon_eventmin) & (ts_lons <= lon_eventmax))
# ts = ts[:, lonst]
# tsfinallon = ts_lons[lonst]

# # select the latitude
# lats = np.arange(-90,91,1)
# lons = np.arange(0,360,1)
# lat_minHW = 0; lat_maxHW = 72; lon_minHW = 130; lon_maxHW = 320
# pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))
# pos_lons = np.where((lons >= lon_minHW) & (lons <= lon_maxHW))
# lats_SST = lats[pos_lats]
# lons_SST = lons[pos_lons]
# Ls = 49
# Lsindex = np.where((lats_SST >= Ls-2) & (lats_SST <= Ls+2))[0]
# # Lsindex = np.where(lats_SST == Ls)[0][0]
# tnp = tnp[:, Lsindex, :]
# tnp = np.nanmean(tnp, axis=1)
# lon_eventmin = 150; lon_eventmax = 270
# lonst = np.where((lons_SST >= lon_eventmin) & (lons_SST <= lon_eventmax))
# tnp = tnp[:, lonst]
# sstfinallon = lons_SST[lonst]
# # calculate the sst anomalies
# timei = []
# doy_list = []  
# start_date = dt.datetime(1981, 9, 1)
# end_date = dt.datetime(2022, 8, 31)
# for ordinal in range(start_date.toordinal(), end_date.toordinal() + 1):
#     current_date = dt.datetime.fromordinal(ordinal)
#     date_str = current_date.strftime('%m%d') + str(current_date.year).zfill(4)
#     timei.append(date_str)
#     doy_list.append(current_date.timetuple().tm_yday)
# doy = doy_list # get DOY
# unique_doy = np.unique(doy)  # get the only DOY
# climatology = np.empty((len(unique_doy), tnp.shape[1], tnp.shape[2]))
# Tanomly = np.empty_like(tnp)
# tnp95 = np.empty_like(tnp)
# for i, day in enumerate(unique_doy):
#     day_indices = np.where(doy == day)[0]
#     climatology[i,:,:] = np.nanmean(tnp[day_indices,:,:], axis=0)
#     Tanomly[day_indices,:,:] = tnp[day_indices,:,:] - climatology[i,:,:]
#     tnp95[day_indices,:,:] = np.nanpercentile(tnp[day_indices,:,:] - climatology[i,:,:], 90, axis=0) # get the 95th percentile
# tnp = Tanomly 
# MHW_flag = tnp > tnp95

# # calculate the ts anomalies
# timei = []
# doy_list = []  
# start_date = dt.datetime(1950, 1, 1)
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
# ts95 = np.empty_like(ts)
# for i, day in enumerate(unique_doy):
#     day_indices = np.where(doy == day)[0]
#     climatology[i,:,:] = np.nanmean(ts[day_indices,:,:], axis=0)
#     Tanomly[day_indices,:,:] = ts[day_indices,:,:] - climatology[i,:,:]
#     ts95[day_indices,:,:] = np.nanpercentile(ts[day_indices,:,:] - climatology[i,:,:], 95, axis=0) # get the 95th percentile
# ts = Tanomly
# LHW_flag = ts > ts95

# # get the target time
# t = np.arange(date(1981,9,1).toordinal(),date(2022,8,31).toordinal()+1) # the target time
# stday = date(2021,6,10).toordinal()
# edday = date(2021,7,5).toordinal()
# stindex = np.where(t == stday)[0][0]
# edindex = np.where(t == edday)[0][0]
# tnpevent = tnp[stindex:edindex+1,0, :]
# MHW_flag = MHW_flag[stindex:edindex+1,0, :]
# print('MHW tnp and flag shapes:')
# print(tnpevent.shape)
# print(MHW_flag.shape)

# t = np.arange(date(1950,1,1).toordinal(),date(2021,12,31).toordinal()+1) # the target time
# stday = date(2021,6,10).toordinal()
# edday = date(2021,7,5).toordinal()
# stindex = np.where(t == stday)[0][0]
# edindex = np.where(t == edday)[0][0]
# datearr = [date.fromordinal(day) for day in range(stday, edday + 1)]
# print(datearr)
# tsevent = ts[stindex:edindex+1,0, :]
# LHW_flag = LHW_flag[stindex:edindex+1,0, :]
# print('LHW tnp and flag shapes:')
# print(tsevent.shape)
# print(LHW_flag.shape)

# print('Step 2 finished')


# np.save('/scratch/bell/hu1029/tsfinallon_5dgavg.npy', tsfinallon)
# np.save('/scratch/bell/hu1029/sstfinallon_5dgavg.npy', sstfinallon)
# np.save('/scratch/bell/hu1029/lwafinallon_5dgavg.npy', lwafinallon)
# np.save('/scratch/bell/hu1029/datearr_5dgavg.npy', datearr)
# np.save('/scratch/bell/hu1029/tsevent_5dgavg.npy', tsevent)
# np.save('/scratch/bell/hu1029/tnpevent_5dgavg.npy', tnpevent)
# np.save('/scratch/bell/hu1029/LWAevent_5dgavg.npy', LWAevent)
# np.save('/scratch/bell/hu1029/LHW_flag_5dgavg.npy', LHW_flag)
# np.save('/scratch/bell/hu1029/MHW_flag_5dgavg.npy', MHW_flag)


# Load the data
tsfinallon = np.load('/scratch/bell/hu1029/tsfinallon_5dgavg.npy', allow_pickle=True)
sstfinallon = np.load('/scratch/bell/hu1029/sstfinallon_5dgavg.npy', allow_pickle=True)
lwafinallon = np.load('/scratch/bell/hu1029/lwafinallon_5dgavg.npy', allow_pickle=True)
datearr = np.load('/scratch/bell/hu1029/datearr_5dgavg.npy', allow_pickle=True)
tsevent = np.load('/scratch/bell/hu1029/tsevent_5dgavg.npy', allow_pickle=True)
tnpevent = np.load('/scratch/bell/hu1029/tnpevent_5dgavg.npy', allow_pickle=True)
LWAevent = np.load('/scratch/bell/hu1029/LWAevent_5dgavg.npy', allow_pickle=True)
LHW_flag = np.load('/scratch/bell/hu1029/LHW_flag_5dgavg.npy', allow_pickle=True)
MHW_flag = np.load('/scratch/bell/hu1029/MHW_flag_5dgavg.npy', allow_pickle=True)

print(LHW_flag[:,-1])
print(LHW_flag[-1,:])
print(MHW_flag[:,-1])
print(MHW_flag[-1,:])

print(datearr)
plt.figure(figsize=(12, 6))
    # US ts
tslevel = np.arange(0, 24, 1)
contour_ts = plt.contourf(tsfinallon, datearr, tsevent, levels=30, cmap='plasma_r')
plt.colorbar(contour_ts, label='TS anomaly (°C)')  
plt.contourf(tsfinallon, datearr, LHW_flag, levels=[0.5, 1], colors='none', hatches=['+', ''], alpha=0)
    # MHW
tnpevent = tnpevent[:,:-10]
print(np.nanmax(tnpevent))
sstfinallon = sstfinallon[:-10]
MHW_flag = MHW_flag[:,:-10]
mhwlevel = np.arange(-3.2, 3.3, 0.1)
contour_sst = plt.contourf(sstfinallon, datearr, tnpevent, levels=30, cmap='coolwarm')
plt.colorbar(contour_sst, label='SST anomaly (°C)')
plt.contourf(sstfinallon, datearr, MHW_flag, levels=[0.5, 1], colors='none', hatches=['+', ''], alpha=0)
    # LWA
# im3 = plt.contour(lwafinallon, datearr, LWAevent, levels=10, colors='green', linewidths=0.8)
# plt.clabel(im3, inline=True, fontsize=8, fmt='%1.1f')
im_pos = plt.contour(
    lwafinallon, datearr, LWAevent,
    levels=np.arange(0, 240, 60),  # >0 
    colors='green',
    linewidths=1.4,
    linestyles='solid'
)
plt.clabel(im_pos, inline=True, fontsize=8, fmt='%1.1f')

im_neg = plt.contour(
    lwafinallon, datearr, LWAevent,
    levels=np.arange(-300, 0, 60),  # <0 
    colors='green',
    linewidths=0.8,
    linestyles='dashed'
)
plt.clabel(im_neg, inline=True, fontsize=8, fmt='%1.1f')
ax = plt.gca()
ax.yaxis_date()  
ax.yaxis.set_major_locator(mdates.DayLocator(interval=4))
ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.xlabel('Longitude')
plt.ylabel('Date')
plt.title('Havmoller Diagram_SST(49N)_TS(49N)_LWA(49N)')

# Save and display the plot
plt.savefig('HavmollerDiagram49N_SST_TS_anomaly_withHWlabel_withZ500anom_single49_hatches_MHW90th_5dgAveraged.png')
plt.show()

print('done')


