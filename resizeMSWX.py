import torch
import numpy as np
from multiprocessing import Pool
import netCDF4
from netCDF4 import Dataset
import os
import numba
from numba import jit
import warnings
warnings.filterwarnings("ignore")


gpu = '0,1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
def read_MSWX_npy(data_dir):
    data = np.load(data_dir)
    if 'Temp' in data_dir or 'Tmax' in data_dir or 'Tmin' in data_dir:
        data = T_z(np.squeeze(data))
    return np.squeeze(data)

#################################################################################################
def find_filename(root_dir, time_range = [1979,2021]):
    files = sorted(os.listdir(root_dir))
    file_list = []
    for file in files:
        if int(file[0:4]) >= time_range[0] and int(file[0:4]) <= time_range[1]:
            file_list.append(root_dir+file)
    return file_list

def find_all_filename(MSWX_list):
    result_list = find_filename(MSWX_list)
    return result_list
    
################################################################################################
import cv2

def corase(data_dir):
    data0 = np.flipud(read_MSWX_npy(data_dir))
    ####
    row,col = data0.shape
    data_c = np.zeros((int(row//10),int(col//10)))
    for i in range(int(row//10)):
        for j in range(int(col//10)):
            if i ==int(row//10-1):
                if j == int(col//10-1):
                    data_c[i,j] = np.nanmean(data0[i*10-5:i*10,j*10-5:j*10])
                elif j == 0:
                    data_c[i,j] = np.nanmean(data0[i*10-5:i*10,j*10:j*10+5])
                else:
                    data_c[i,j] = np.nanmean(data0[i*10-5:i*10,j*10-5:j*10+5])
            elif i == 0:
                if j == 0:
                    data_c[i,j] = np.nanmean(data0[i*10:i*10+5,j*10:j*10+5])
                else:
                    data_c[i,j] = np.nanmean(data0[i*10:i*10+5,j*10-5:j*10+5])
            elif j == int(col//10-1) and i != int(row//10-1) and i != 0:
                data_c[i,j] = np.nanmean(data0[i*10-5:i*10+5,j*10-5:j*10])
            elif j == 0 and i != 0 and i != int(row//10-1):
                data_c[i,j] = np.nanmean(data0[i*10-5:i*10+5,j*10:j*10+5])
            else:
                data_c[i,j] = np.nanmean(data0[i*10-5:i*10+5,j*10-5:j*10+5])
    ####
    save_dir = 'path to save coarse results'
    print(save_dir[-4:]+data_dir[-11:-3]+' c  Done  ' + str(data_c.shape))
    np.save(save_dir+data_dir[-11:-3]+'npy',data_c)
    
    ####
    data = cv2.resize(data_c,(col,row),interpolation=cv2.INTER_LINEAR)
    if 'PRE' in data_dir:
        save_dir = 'path to save double linear result'
        
    print(save_dir[-4:]+data_dir[-11:-3]+' linear  Done  ' + str(data.shape))
    np.save(save_dir+data_dir[-11:-3]+'npy',data)
    ####

#################################################################################################
#################################################################################################
if __name__ == '__main__':

    pool = Pool(processes = 12)

    MSWX_root_dir = 'MSWX data path'
    MSWX_list = [MSWX_root_dir]
    all_listsss = pool.map(find_all_filename,MSWX_list)
    for i in range(len(all_listsss)):
        pool.map(corase,all_listsss[i])