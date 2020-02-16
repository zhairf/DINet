#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:12:10 2020

@author: diruifeng
"""

import h5py
import numpy as np
import os
import provider

#f = os.path.exists('data/modelnet40_ply_hdf5_2048/ply_data_train0.h5')
#print(f)
#err_num = 5708
#batch_size = 128
def make_ModelNet_data_A_B(err_num,batch_size):
    ModelNet_TRAIN_FILES = provider.getDataFiles('../data/modelnet40_ply_hdf5_2048/train_files.txt')
    
    class_data = {l:[] for l in range(40)}
    for data_set in ModelNet_TRAIN_FILES:
        file = h5py.File('..//'+data_set,'r')
        data = file['data'][...]
        label = np.reshape(file['label'][...],-1)
        normal = file['normal'][...]
        for i in range(data.shape[0]):
            class_data[label[i]].append(data[i])
        file.close()
        
    for i in range(40):
        class_data[i] = np.array(class_data[i])
        
    j = 0
    for data_set in ModelNet_TRAIN_FILES:
        print('making ModelNet_A_B_'+str(j)+'.h5')
        file = h5py.File('..//'+data_set,'r')
        data_A = file['data'][...]
        label_A = file['label'][...]
        
        data_A, label_A, _ = provider.shuffle_data(data_A, label_A)  
        ERR = err_num * int(data_A.shape[0]/9843)
        data_B, label_B = provider.get_data_with_err_ModelNet(label_A,class_data,err_num)
        label_B = np.reshape(label_B, [-1,1])
        data_A_B = np.zeros([2*data_A.shape[0],2048,3])
        label_A_B = np.zeros([2*data_A.shape[0],1])
        
        for i in range(int(data_A.shape[0]/(batch_size/2))):
            data_A_B[int((i+0)*batch_size):int((i+0.5)*batch_size)] = data_A[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
            data_A_B[int((i+0.5)*batch_size):int((i+1)*batch_size)] = data_B[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
            label_A_B[int((i+0)*batch_size):int((i+0.5)*batch_size)] = label_A[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
            label_A_B[int((i+0.5)*batch_size):int((i+1)*batch_size)] = label_B[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
        
        data_set_A_B = h5py.File('data/ModelNet_A_B_'+str(j)+'.h5','w')
        data_set_A_B['data'] = data_A_B
        data_set_A_B['label'] = label_A_B
        data_set_A_B.close()
        file.close()
        j += 1
        
def make_ScanObjectNN_data_A_B(err_num,batch_size):
    class_data = {l:[] for l in range(15)}
    file = h5py.File('../data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5','r')
    data_A = file['data'][...]
    label = np.reshape(file['label'][...],-1)
    for i in range(data_A.shape[0]):
        class_data[label[i]].append(data_A[i])
        
    for i in range(15):
        class_data[i] = np.array(class_data[i])
        
    print('making ScanObject_A_B.h5')
    label_A = np.reshape(label,[-1,1])
    data_A, label_A, _ = provider.shuffle_data(data_A, label_A)  
    ERR = err_num
    data_B, label_B = provider.get_data_with_err_ScanObjectNN(label_A,class_data,ERR)
    label_B = np.reshape(label_B, [-1,1])
    data_A_B = np.zeros([int((2*data_A.shape[0])/batch_size)*batch_size,2048,3])
    label_A_B = np.zeros([int((2*data_A.shape[0])/batch_size)*batch_size,1])
    
    for i in range(int(data_A.shape[0]/(batch_size/2))):
        data_A_B[int((i+0)*batch_size):int((i+0.5)*batch_size)] = data_A[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
        data_A_B[int((i+0.5)*batch_size):int((i+1)*batch_size)] = data_B[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
        label_A_B[int((i+0)*batch_size):int((i+0.5)*batch_size)] = label_A[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
        label_A_B[int((i+0.5)*batch_size):int((i+1)*batch_size)] = label_B[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
    
    data_set_A_B = h5py.File('data/ScanObjectNN_A_B.h5','w')
    data_set_A_B['data'] = data_A_B
    data_set_A_B['label'] = label_A_B
    data_set_A_B.close()
    file.close()

def make_ScanObjectNN_data_batch_A_B(err_num,batch_size):   
    class_data = {l:[] for l in range(15)}
    file = h5py.File('../data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5','r')
    data_A = file['data'][...]
    label = np.reshape(file['label'][...],-1)
    for i in range(data_A.shape[0]):
        class_data[label[i]].append(data_A[i])
        
    for i in range(15):
        class_data[i] = np.array(class_data[i])
        
    print('making ScanObject_A_B.h5')
    
    label_A = np.reshape(label,[-1,1])
    data_A, label_A, _ = provider.shuffle_data(data_A, label_A)  
    ERR = int(err_num/batch_size)
    
    data_B = np.zeros([int((data_A.shape[0])/batch_size)*batch_size,2048,3])
    label_B = np.zeros([int((data_A.shape[0])/batch_size)*batch_size])
    for i in range(int(label_A.shape[0]/(batch_size/2))):
        data_B[int(i*(batch_size/2)):int((i+1)*(batch_size/2))], \
        label_B[int(i*(batch_size/2)):int((i+1)*(batch_size/2))] = \
        provider.get_data_with_err_ScanObjectNN(label_A[int(i*(batch_size/2)):int((i+1)*(batch_size/2))],class_data,ERR)
        
    label_B = np.reshape(label_B, [-1,1])
    data_A_B = np.zeros([int((2*data_A.shape[0])/batch_size)*batch_size,2048,3])
    label_A_B = np.zeros([int((2*data_A.shape[0])/batch_size)*batch_size,1])
    
    for i in range(int(data_A.shape[0]/(batch_size/2))):
        data_A_B[int((i+0)*batch_size):int((i+0.5)*batch_size)] = data_A[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
        data_A_B[int((i+0.5)*batch_size):int((i+1)*batch_size)] = data_B[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
        label_A_B[int((i+0)*batch_size):int((i+0.5)*batch_size)] = label_A[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
        label_A_B[int((i+0.5)*batch_size):int((i+1)*batch_size)] = label_B[int((i+0)*(batch_size/2)):int((i+1)*(batch_size/2))]
    
    data_set_A_B = h5py.File('data/ScanObjectNN_A_B.h5','w')
    data_set_A_B['data'] = data_A_B
    data_set_A_B['label'] = label_A_B
    data_set_A_B.close()
    file.close()

#make_data_A_B(5000,32)
#make_ScanObjectNN_data_batch_A_B(5000,128)
