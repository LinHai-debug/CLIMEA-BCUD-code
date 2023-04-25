import numpy as np
import netCDF4 as nc
import h5py
import math
import shutil
import os
import cv2
from multiprocessing import Pool
import sys


def Leap_Year_If(year):
    '''return 1 means leap year,return 0 means not leap year.'''
    if (year%4==0 and year%100!=0) or year%400==0:
        return 1
    return 0

def date_series(year_start, year_end):
    years = [str(i) for i in range(year_start,year_end+1)]
    days = []
    for i in range(1,366):
        if i < 10:
            days.append('00'+str(i))
        elif i >= 10 and i < 100:
            days.append('0'+str(i))
        else:
            days.append(str(i))
    
    dates = []
    for year in years:
        for day in days:
            dates.append(year+day)
        if Leap_Year_If(int(year)):###
            dates.append(year+'366')###
    return dates

def read_nc(year_start,d = 49):    
    variables = ['huss']
    for variable in variables:
        if d == 6:
            year_end = year_start+d
            timed = str(year_start)+'0101-' + str(year_end) + '1231'
        elif d == 5:
            year_end = year_start+5
            timed = str(year_start)+'0101-' + str(year_end) + '1231'
        elif d == 4:
            year_end = year_start+4
            timed = str(year_start)+'0101-' + str(year_end) + '1231'
        else:
            year_end = year_start+d
            timed = str(year_start)+'0101-' + str(year_end) + '1231'
        file_path = data_dir + variable+'_day_'+model+'_'+ssp+'_r1i1p1f1_gn_'+timed+'.nc'
        nc_obj = nc.Dataset(file_path)

        lat=(nc_obj.variables['lat'][:])
        lon=(nc_obj.variables['lon'][:])
        prcp=(nc_obj.variables[variable][:])
        prcp[prcp == nc_obj.variables[variable]._FillValue] = np.nan
        #prcp *= 86400#################################################################################################
        lat_epsilon = abs(lat[lat.shape[0]//2]-lat[lat.shape[0]//2-1])
        lon_epsilon = abs(lon[lon.shape[0]//2]-lon[lon.shape[0]//2-1])

        for i in range(lat.shape[0]):
            if abs(lat[i] - 4.95) <= lat_epsilon+0.01:
                start_lat = i
                print(lat[i])
                break
        for j in range(lat.shape[0]):
            i = lat.shape[0]-j-1
            if abs(lat[i] - 60.05) <= lat_epsilon+0.01:
                end_lat = i
                print(lat[i])
                break
        for i in range(lon.shape[0]):
            if abs(lon[i] - 64.75) <= lon_epsilon+0.01:
                start_lon = i
                print(lon[i])
                break
        for j in range(lon.shape[0]):
            i = lon.shape[0]-j-1
            if abs(lon[i] - 150.25) <= lon_epsilon+0.01:
                end_lon = i
                print(lon[i])
                break

        pr = prcp[:,start_lat:end_lat,start_lon:end_lon]
        
        if year_end == 2015 or year_end == 2101:
            year_end -= 1
        dates = date_series(year_start, year_end)
        sample,row,col = pr.shape
        print(sample,len(dates))
        if sample != len(dates):
            print('Error!')
        save_dir = 'save path (.npy)'
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
        for i,date in enumerate(dates):
            data0 = cv2.resize(pr[i,:,:],(85,55),interpolation=cv2.INTER_LINEAR)
            np.save(save_dir+date+'.npy',data0)
            print(str(lat[start_lat]),str(lat[end_lat]),str(lon[start_lon]),str(lon[end_lon]))
            print(date + 'Done!',np.max(data0),np.min(data0),np.mean(data0),data0.shape)

if __name__ == '__main__':
    model = sys.argv[-2]
    data_dir = 'CMIP6 data path (.nc)'
    ssp = sys.argv[-1]
    pool = Pool(processes = 6)
    if 'ssp' in ssp:
        year_starts = [i for i in range(2015,2065,50)]
        pool.map(read_nc,year_starts)
        read_nc(2065,d = 35)
    else:
        year_starts = [i for i in range(1950,2000,50)]
        pool.map(read_nc,year_starts)
        read_nc(2000,d = 14)
    
