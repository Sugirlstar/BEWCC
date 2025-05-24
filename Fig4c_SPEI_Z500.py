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

# function
def anomalies_seasons(df_VAR):
    import numpy as np
    import pandas as pd
    import datetime as dt
    ANOMA = df_VAR * np.nan

    def custom_strptime(time_data):
        if len(time_data) == 7:
            month = time_data[:2]
            day = time_data[2:4]
            year = '0' + time_data[4:]
            date_str = f'{month}{day}{year}'
            return dt.datetime.strptime(date_str, '%m%d%Y')
        else:
            return dt.datetime.strptime(time_data, '%m%d%Y')
    
    # dates_d = np.array([dt.datetime.strptime(iii, '%m%d%Y') for iii in df_VAR.index])
    dates_d = np.array([custom_strptime(iii) for iii in df_VAR.index])
    # 遍历每个月份，从1到12月：
    for i in np.arange(1, 13):
        mes = df_VAR.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]] #筛选出 df_VAR 中所有的第i个月，并存储在 mes 中
        dates_d_mes = dates_d[np.where(np.array([ii.month for ii in dates_d])==i)[0]] #筛选出 dates_d 中属于当前月份的日期，并存储在 dates_d_mes 中
        temp = mes * np.nan
        Nodays = np.array([ii.day for ii in dates_d_mes]).max()
        if np.isnan(Nodays) == True: continue
        # 循环遍历当前月份的每一天，从1号到最大天数
        for j in np.arange(1, Nodays + 1):
            # 筛选出 mes 中属于当前天的数据，并存储在 dia 中，也就是第i个月第j天，所有年份的
            dia = mes.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]]
            media = dia.mean() # 求第i个月第j天的均值
            anoma = dia - media
            temp.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]] = anoma
        ANOMA.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]] = temp
    return ANOMA

# read in SPEI
nc_name = '/scratch/bell/hu1029/Data/raw/Global_dailySPEI90/Daily_SPEI_2021_90Day.nc'
ncfile = xr.open_dataset(nc_name)
SManom = np.array(ncfile.spei)
SManom = np.transpose(SManom, (0, 2, 1))
print(SManom.shape)
lat = np.array(ncfile.lat)
lon = np.array(ncfile.lon)
lon = lon + 180
print(lon)
    # get the region
Ls = 49
Lsindex = np.abs(lat - Ls).argmin()
print(Lsindex)
SManom = SManom[:, Lsindex, :]
print(SManom.shape)
lon_eventmin = 235; lon_eventmax = 270
lonst = np.where((lon >= lon_eventmin) & (lon <= lon_eventmax))[0]
SManom = SManom[:, lonst]
smfinallon = lon[lonst]
print(SManom.shape)
SManom[SManom == -9999] = np.nan

    # get the target time
t = np.arange(date(2021,1,1).toordinal(),date(2021,12,31).toordinal()+1) # the target time
stday = date(2021,6,15).toordinal()
edday = date(2021,7,5).toordinal()
stindex = np.where(t == stday)[0][0]
edindex = np.where(t == edday)[0][0]
SManomevent = SManom[stindex:edindex+1,:]
print(SManomevent.shape)

# read in Z500
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
Zanom_daily = ds['z'].squeeze().resample(time='1D').mean()
Zanom = np.array(Zanom_daily)
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
print(lat, flush=True)
print(lon, flush=True)
print(Zanom.shape, flush=True)
print('-------- Zanom loaded --------', flush=True)
# time for Z500
timei = []
start_date = dt.datetime(1979, 1, 1)
end_date = dt.datetime(2021, 12, 31)
for ordinal in range(start_date.toordinal(), end_date.toordinal() + 1):
    current_date = dt.datetime.fromordinal(ordinal)
    date_str = current_date.strftime('%m%d') + str(current_date.year).zfill(4)
    timei.append(date_str)
print(len(timei))
# get the region
Ls = 49
Lsindex = np.where(lat == Ls)[0][0]
print(Lsindex)
LWA_td = Zanom[:, Lsindex, :]
lon_eventmin = 235; lon_eventmax = 270
lonst = np.where((lon >= lon_eventmin) & (lon <= lon_eventmax))[0]
LWA_td = LWA_td[:, lonst]
lwafinallon = lon[lonst]
print(LWA_td.shape)

    # get the target time
t = np.arange(date(1979,1,1).toordinal(),date(2021,12,31).toordinal()+1) # the target time
stday = date(2021,6,15).toordinal()
edday = date(2021,7,5).toordinal()
stindex = np.where(t == stday)[0][0]
edindex = np.where(t == edday)[0][0]
LWAevent = LWA_td[stindex:edindex+1,:]
print(LWAevent.shape)

datearr = [date.fromordinal(day) for day in range(stday, edday + 1)]

print(np.nanmax(SManomevent))
print(np.nanmin(SManomevent))

plt.figure(figsize=(8, 7))
    # rzSManom
levels = np.arange(-1.8, 1.9, 0.1)
contour_ts = plt.contourf(smfinallon, datearr, SManomevent, levels=levels, cmap='BrBG')
plt.colorbar(contour_ts, label='SPEI')  
    # LWA
all_levels = np.arange(-225, 226, 25)
lwa_levels = all_levels[np.abs(all_levels) >= 50]
im3 = plt.contour(lwafinallon, datearr, LWAevent, levels=lwa_levels, colors='black', linewidths=0.8)
plt.clabel(im3, inline=True, fontsize=8, fmt='%1.1f')

plt.xlabel('Longitude')
plt.ylabel('Date')
plt.title('Havmoller Diagram_LWA(49N)_SPEI(49N)')

# Save and display the plot
plt.savefig('Havmoller Diagram_Z500(49N)_SPEI(49N).png')
plt.show()

print('done')


