#import torch
import numpy as np
from multiprocessing import Pool
#import multiprocessing
import netCDF4
from netCDF4 import Dataset
import os
import numba
from numba import jit
import warnings
import sys

import tensorflow as tf


warnings.filterwarnings("ignore")


os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[-2]



import keras
import keras.backend as K
 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
 
import keras.backend as K

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True



def custom_loss(y_true,y_pred):
    y_true = tf.where(tf.greater(y_true,0),y_true*weight,y_true)
    y_pred = tf.where(tf.greater(y_true,0),y_pred*weight,y_pred)
    return tf.reduce_mean(tf.abs(y_true-y_pred))


def read_MSWX_npy(data_dir):
    data = np.load(data_dir)
    if 'Temp' in data_dir or 'MAX' in data_dir or 'MIN' in data_dir:
        data = T_z(np.squeeze(data))
    return np.squeeze(data)

#################################################################################################
def find_filename(root_dir, time_range = [int(sys.argv[-5]),int(sys.argv[-5])+15]):
    files = sorted(os.listdir(root_dir))
    file_list = []
    
    for file in files:
        a = int(sys.argv[-4])
        if int(file[0:4]) >= time_range[0] and int(file[0:4]) <= time_range[1]:
            if sys.argv[-3] == 'mam' and int(file[4:7]) >= a and int(file[4:7]) < 151 and int(file[4:7]) < a+1:
                file_list.append(root_dir+file)
            elif sys.argv[-3] == 'jja' and int(file[4:7]) >= a and int(file[4:7]) < 243 and int(file[4:7]) < a+1:
                file_list.append(root_dir+file)
            elif sys.argv[-3] == 'son' and int(file[4:7]) >= a and int(file[4:7]) < 334 and int(file[4:7]) < a+1:
                file_list.append(root_dir+file)
            elif sys.argv[-3] == 'djf' and (int(file[4:7]) >= 334 or int(file[4:7]) < 59) and int(file[4:7]) >= a and int(file[4:7]) < a+1:
                file_list.append(root_dir+file)
    #print(file_list)
    return file_list

def find_all_filename(MSWX_list):
    result_list = find_filename(MSWX_list)
    return result_list
    
################################################################################################
#################################################################################################
def Predict(data_dir,model):


    data = np.squeeze(read_MSWX_npy(data_dir))
    mean = np.nanmean(data)
    std = np.nanstd(data)
    data = (data-mean)/(1e-6+std)
    
    dem = np.squeeze(np.load('elevation data path'))
    dem = (dem-np.nanmean(dem))/(np.nanstd(dem)+1e-6)
    
    
    
    save_dir = 'path to save downscaling results' 
        
        
    data1 = np.zeros((1,552,856,2))
    data1[0,:,:,0] = data
    data1[0,:,:,1] = dem

    predict = model.predict(x=tf.data.Dataset.from_tensors(data1))#加快速度2
    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
    print(save_dir[-4:]+data_dir[-11:-4]+'  Done  ')

    predict = std * predict + mean

    np.save(save_dir+data_dir[-11:-4]+'.npy',predict[0,:,:,0])


#################################################################################################
if __name__ == '__main__':

    #pool = Pool(processes = 6)
    MSWX_root_dir = 'MSWX double linear path'
    MSWX_list = MSWX_root_dir + sys.argv[-1] + '/'
    all_listsss = find_all_filename(MSWX_list)
    
    pb_file_path = 'UNet model path'
    weight = int(sys.argv[-7])
    model = tf.keras.models.load_model(pb_file_path)
    for all_list in all_listsss:
        Predict(all_list, model)