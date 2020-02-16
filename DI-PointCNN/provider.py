import os
import sys
import numpy as np
import h5py
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
'''
# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))
'''

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def get_num_data(contrast_data,num):
    #返回对用类别点云全部数据
    return contrast_data[num][:]
    

def random_get_contrast(data):
    #返回一张数据
    total_num = data.shape[0]
    one_point_cloud = data[random.randint(0,total_num-1)]
    return one_point_cloud

def load_contrast_data(filename):
    f = h5py.File(filename,'r')   #打开h5文件  
    return f

def get_contrast_data(label,DATA):#DATA是整体对照数据    用这个！！！！
    contrast_data = []
    for num in label:
        category_data = get_num_data(DATA,num)
        one_contrast_data = random_get_contrast(category_data)
        contrast_data.append(one_contrast_data)
    contrast_data = np.float32(contrast_data)
    return contrast_data

def get_contrast_data_eval(label,DATA):#DATA是整体对照数据
    contrast_data = []
    for line in label:
        for num in line:
            category_data = get_num_data(DATA,num)
            one_contrast_data = random_get_contrast(category_data)
            contrast_data.append(one_contrast_data)
    contrast_data = np.float32(contrast_data)
    contrast_data = np.reshape(contrast_data,[label.shape[0],label.shape[1],2048,3])
    return contrast_data #[label,batch,point,xyz]
    
def shuffle_data_2B(data_A, labels_A, data_B, labels_B):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels_A))
    np.random.shuffle(idx)
    return data_A[idx, ...], labels_A[idx], data_B[idx, ...], labels_B[idx]
    
def get_error_label(label):
    err_label = label.copy()
    probability=[0.0637,0.0108,0.0523,0.0176,0.0581,0.034,0.0065,0.02,0.0903,0.017,0.008,0.0139,0.0203,0.0111,0.0203,0.0151,0.0174,0.0158,0.0147,0.0126,0.0151,0.0289,0.0473,0.0203,0.0089,0.0235,0.0243,0.0106,0.0117,0.013,0.0691,0.0126,0.0091,0.0398,0.0166,0.035,0.0271,0.0483,0.0088,0.0105]
    i = 0
    for one in label:
        while True:
            err_label[i] = np.random.choice(a=40, size=1, replace=True,p=probability)
            if err_label[i] != one:
                break
        i +=1
    return label

def get_one_error_label_MoedlNet(label):
    err_label = label.copy()
    probability=[0.0637,0.0108,0.0523,0.0176,0.0581,0.034,0.0065,0.02,0.0903,0.017,0.008,0.0139,0.0203,0.0111,0.0203,0.0151,0.0174,0.0158,0.0147,0.0126,0.0151,0.0289,0.0473,0.0203,0.0089,0.0235,0.0243,0.0106,0.0117,0.013,0.0691,0.0126,0.0091,0.0398,0.0166,0.035,0.0271,0.0483,0.0088,0.0105]
    while True:
        err_label = np.random.choice(a=40, size=1, replace=True,p=probability)
        if err_label != label:
            break
    return err_label

def get_one_error_label_ScanObjectNN(label):
    err_label = label.copy()
    while True:
        err_label = np.random.choice(a=15, size=1, replace=True)
        if err_label != label:
            break
    return err_label

def get_data_with_err_ModelNet(current_label,contrast_DATA,num):
    files = len(current_label)
    label_with_err = current_label.copy()
    for i in range(num):
        idx = random.randint(0,files-1)
        label_with_err[idx] = get_one_error_label_MoedlNet(label_with_err[idx])
    label_with_err = np.squeeze(label_with_err)
    data = get_contrast_data(label_with_err,contrast_DATA) 
    return data, label_with_err

def get_data_with_err_ScanObjectNN(current_label,contrast_DATA,num):
    files = len(current_label)
    label_with_err = current_label.copy()
    for i in range(num):
        idx = random.randint(0,files-1)
        label_with_err[idx] = get_one_error_label_ScanObjectNN(label_with_err[idx])
    label_with_err = np.squeeze(label_with_err)
    data = get_contrast_data(label_with_err,contrast_DATA) 
    return data, label_with_err

def get_label(current_label,num):
    files = len(current_label)
    label_with_err = current_label.copy()
    for i in range(num):
        idx = random.randint(0,files-1)
        label_with_err[idx] = get_one_error_label(label_with_err[idx])
    label_with_err = np.squeeze(label_with_err)
    return label_with_err
    