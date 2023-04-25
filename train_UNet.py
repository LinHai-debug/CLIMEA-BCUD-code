import sys
import time
import argparse
from glob import glob
import os
import h5py
import numpy as np
from random import shuffle
from tensorflow import keras
from PIL import Image
#import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


# custom tools
sys.path.insert(0, './utils')
from namelist import *
import model_utils as mu
import train_utils as tu

weight = int(sys.argv[-5])

def custom_loss(y_true,y_pred):
    y_true = tf.where(tf.greater(y_true,0),y_true*weight,y_true)
    y_pred = tf.where(tf.greater(y_true,0),y_pred*weight,y_pred)
    return tf.reduce_mean(tf.abs(y_true-y_pred))

def read_input_list(file_list):
    data = []
    for file in file_list:
        d0 = np.squeeze(np.load(file))
        mean = np.nanmean(d0)
        std = np.nanstd(d0)
        data.append((np.squeeze(np.load(file))-mean)/(std+1e-6))
    return np.array(data)

def read_label_list(file_list):
    data = []
    for file in file_list:
        d0 = np.flipud(np.squeeze(np.load(file)))
        mean = np.nanmean(d0)
        std = np.nanstd(d0)
        data.append((np.flipud(np.squeeze(np.load(file)))-mean)/(std+1e-6))
    return np.array(data)

def find_filename(root_dir, season, time_range = [1979,2018]):
    files = sorted(os.listdir(root_dir))
    file_list = []
    for file in files:
        #print(file[0:4])
        if int(file[0:4]) >= time_range[0] and int(file[0:4]) <= time_range[1]:
            if season == 'mam' and int(file[4:7]) >= 59 and int(file[4:7]) < 151:
                file_list.append(root_dir+file)
            elif season == 'jja' and int(file[4:7]) >= 151 and int(file[4:7]) < 243:
                file_list.append(root_dir+file)
            elif season == 'son' and int(file[4:7]) >= 243 and int(file[4:7]) < 334:
                file_list.append(root_dir+file)
            elif season == 'djf' and (int(file[4:7]) >= 334 or int(file[4:7]) < 59):
                file_list.append(root_dir+file)
    return file_list
    
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[-1]
model_name = sys.argv[-2]#model name: Unet or UNet-AE or Nest-Unet
variables = sys.argv[-3] #variables
znz = sys.argv[-4] #z or noz
sample_per_batch = 1
lr = 5e-5
epochs = 101
activation='relu'
pool=False 

if variables == 'PRE':
    N = [64, 96, 128, 160]#做降水
else:
    N = [56, 112, 224, 448]#做温度

if znz == 'z':
    z = np.squeeze(np.load('elevation data path'))
    z  = (z - np.nanmean(z))/(np.nanstd(z)+1e-6)##################################################################
    N_input = 2
else:
    z = np.zeros((2,2))+1
    N_input = 1

if int(sys.argv[-6]) == 0:
    seasons = ['mam','jja','son','djf']#season
elif int(sys.argv[-6]) == 1:
    seasons = ['mam','jja']#season
elif int(sys.argv[-6]) == 2:
    seasons = ['son','djf']#season
    
counts = 3

for season in seasons:
    input_dir = 'inputs path'
    train_input_lists = find_filename(input_dir, season, time_range = [1979,2006])
    valid_input_lists = find_filename(input_dir, season, time_range = [2007,2010])
    
    label_dir = 'labels path'
    train_label_lists = find_filename(label_dir, season, time_range = [1979,2006])
    valid_label_lists = find_filename(label_dir, season, time_range = [2007,2010])
    
    c = list(zip(train_input_lists,train_label_lists))
    
    for i in range(counts):
        save_dir = 'save path'
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
        
        if model_name == 'Unet':
            dscale_u = mu.UNET(N, (None, None, N_input), pool=pool, activation=activation)
        elif model_name == 'UNet-AE':
            dscale_u = mu.UNET_AE(N, (None, None, N_input), pool=pool, activation=activation)
        elif model_name == 'Nest-Unet':
            dscale_u = mu.XNET(N, (None, None, N_input), pool=pool, activation=activation)
        opt_ = keras.optimizers.Adam(lr=lr)
        dscale_u.compile(loss=custom_loss, optimizer=opt_)
        
        L_train = len(train_input_lists)//sample_per_batch
        L_valid = len(valid_input_lists)
        
        T_LOSS = []
        V_LOSS = []

        
        for epoch in range(epochs):
            save_name = 'UNET_epoch_'+str(epoch)
            save_path = save_dir+save_name+'/'
            hist_path = save_dir+'{}_loss.npy'.format(save_name)
            print('epoch = {}'.format(epoch))
            
            if epoch == 0:
                record = 0
                for j in range(L_valid):
                    simple_valid_input = np.squeeze(np.load(valid_input_lists[j]))
                    simple_valid_input = (simple_valid_input-np.nanmean(simple_valid_input))/(np.nanstd(simple_valid_input)+1e-6)
                    row,col = simple_valid_input.shape
                    simple_valid_label = np.flipud(np.squeeze(np.load(valid_label_lists[j]))).reshape(1,row,col)
                    simple_valid_label = (simple_valid_label-np.nanmean(simple_valid_label))/(np.nanstd(simple_valid_label)+1e-6)
                    if np.mean(z) == 1:
                        simple_valid_data = simple_valid_input.reshape(1,row,col,1)
                    else:
                        simple_valid_data = np.zeros((1,row,col,2))
                        simple_valid_data[0,:,:,0] = simple_valid_input
                        simple_valid_data[0,:,:,1] = z
                    record += custom_loss(simple_valid_label.reshape(1,row,col,1), dscale_u.predict([simple_valid_data]))
                record /= L_valid
            
            shuffle(c)
            train_input_lists,train_label_lists = zip(*c)
            
            for j in range(L_train):
                batch_train_input_lists = train_input_lists[(j-1)*sample_per_batch:sample_per_batch*j-1]
                batch_train_label_lists = train_label_lists[(j-1)*sample_per_batch:sample_per_batch*j-1]
                L_batch_lists = len(batch_train_input_lists)
                
                input = read_input_list(batch_train_input_lists)
                batch_train_label = read_label_list(batch_train_label_lists)
                
                if np.mean(z) == 1:
                    batch_train_input = input.reshape(L_batch_lists,row,col,1)
                else:
                    batch_train_input = np.zeros((L_batch_lists,row,col,2))
                    batch_train_input[:,:,:,0] = input
                    for k in range(L_batch_lists):
                        batch_train_input[k,:,:,1] = z                
                loss_ = dscale_u.train_on_batch(batch_train_input, batch_train_label)
                T_LOSS.append(loss_)
                
            record = 0
            for j in range(L_valid):
                simple_valid_input = np.squeeze(np.load(valid_input_lists[j]))
                simple_valid_input = (simple_valid_input-np.nanmean(simple_valid_input))/(np.nanstd(simple_valid_input)+1e-6)
                simple_valid_label = np.flipud(np.squeeze(np.load(valid_label_lists[j]))).reshape(1,row,col)
                simple_valid_label = (simple_valid_label-np.nanmean(simple_valid_label))/(np.nanstd(simple_valid_label)+1e-6)
                row,col = simple_valid_input.shape
                if np.mean(z) == 1:
                    simple_valid_data = simple_valid_input.reshape(1,row,col,1)
                else:
                    simple_valid_data = np.zeros((1,row,col,2))
                    simple_valid_data[0,:,:,0] = simple_valid_input
                    simple_valid_data[0,:,:,1] = z
                record += custom_loss(simple_valid_label.reshape(1,row,col,1), dscale_u.predict([simple_valid_data]))
            record /= L_valid
            V_LOSS.append(record)
            print('VLOSS..................= {}'.format(record))
            if epoch %5 == 0:
                print('save to: {}'.format(save_path))
                dscale_u.save(save_path)
                
        np.save(save_dir+'VLOSS'+season+'.npy', np.array(V_LOSS))
        np.save(save_dir+'TLOSS'+season+'.npy', np.array(T_LOSS))