import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pickle
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
class TSRecurrentDataset(Dataset):   
    def __init__(self, X, y, seq_len,step,device=None):
        super(TSRecurrentDataset, self).__init__()
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.step=step

    def __len__(self):
        return self.X.shape[0]-self.seq_len-self.step+1#self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        x=self.X[index:index+self.seq_len].T
        y=self.y[index+self.seq_len:index+self.seq_len+self.step,:].T
        return torch.tensor(x),torch.tensor(y)

class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
class TSRecurrentDataset(Dataset):   
    def __init__(self, X, y, seq_len,step,device=None):
        super(TSRecurrentDataset, self).__init__()
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.step=step

    def __len__(self):
        return self.X.shape[0]-self.seq_len-self.step+1#self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        x=self.X[index:index+self.seq_len].T
        y=self.y[index+self.seq_len:index+self.seq_len+self.step,:].T
        return torch.tensor(x),torch.tensor(y)
class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, test_version=1,mode="train",task='Forecast',reduce=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.task=task
        # self.scaler = StandardScaler()
        
        # data = np.load(data_path + "/SWaT_train.npy")
        print(reduce)
        if reduce:
            data = np.load(data_path + "/SWaT_train_V3.npy")
        else:
            data = np.load(data_path + "/SWaT_train.npy")
        # self.scaler.fit(data)
        # data = self.scaler.transform(data)
        test_data = np.load(data_path + '/SWaT_test_V'+str(test_version)+'.npy')
        self.test=test_data
        # self.test = self.scaler.transform(test_data)
        
        self.train = data
        self.val = self.train[-5000:,:]
        # print(self.train.shape,self.test.shape)
        self.test_labels = np.load(data_path + "/SWaT_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print('gt',self.test_labels.shape)
    def __len__(self):
        if self.task=='Forecast':
            if self.mode == "train":
                return self.train.shape[0] - self.win_size-self.step+1
            elif (self.mode == 'val'):
                return self.val.shape[0] - self.win_size-self.step+1
            elif (self.mode == 'test'):
                return self.test.shape[0] - self.win_size-self.step+1
            else:
                return self.test.shape[0] - self.win_size-self.step+1
        elif self.task=='Reconstruct':
            # print(self.train.shape)
            if self.mode == "train":
                return (self.train.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'val'):
                return (self.val.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'test'):
                return (self.test.shape[0] - self.win_size) // self.step + 1
            else:
                return (self.test.shape[0] - self.win_size) // self.win_size + 1
    def __getitem__(self, index):
        # index = index * self.step
        if self.task=='Forecast':
            if self.mode == "train":
                return np.float32(self.train[index:index + self.win_size,:].T), np.float32(self.train[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'val'):
                return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.val[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'test'):
                return np.float32(self.test[index:index + self.win_size]).T,np.float32(self.test[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'thre'):
                return np.float32(self.test[index:index + self.win_size]).T,np.float32(self.test[index+self.win_size:index+self.win_size+self.step,:]).T,np.float32(self.test_labels[index+self.win_size:index+self.win_size+self.step])
        elif self.task=='Reconstruct':
            index = index * self.step
            if self.mode == "train":
                return np.float32(self.train[index:index + self.win_size].T),np.float32(self.train[index:index + self.win_size].T)# np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'val'):
                return np.float32(self.val[index:index + self.win_size]),np.float32(self.val[index:index + self.win_size])# np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'test'):
                return np.float32(self.test[index:index + self.win_size].T),np.float32(self.test[index:index + self.win_size].T)
            # np.float32(
            #         self.test_labels[index:index + self.win_size])
            else:
                return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T,\
                       np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T,  \
                       np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
                    
        # else:
        #     return np.float32(self.test[
        #                       index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
        #         self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
    # def __len__(self):

    #     if self.mode == "train":
    #         return (self.train.shape[0] - self.win_size-self.step+1) // self.step + 1
    #     elif (self.mode == 'val'):
    #         return (self.val.shape[0] - self.win_size) // self.step + 1
    #     elif (self.mode == 'test'):
    #         return (self.test.shape[0] - self.win_size) // self.step + 1
    #     else:
    #         return (self.test.shape[0] - self.win_size) // self.win_size + 1

    # def __getitem__(self, index):
    #     index = index * self.step
    #     if self.mode == "train":
    #         return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    #     elif (self.mode == 'val'):
    #         return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    #     elif (self.mode == 'test'):
    #         return np.float32(self.test[index:index + self.win_size]), np.float32(
    #             self.test_labels[index:index + self.win_size])
    #     else:
    #         return np.float32(self.test[
    #                           index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
    #             self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
class WADIA219SegLoader(object):
    def __init__(self, data_path, win_size, step, test_version=1,mode="train",task='Forecast',reduce=False,data_scale=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.task=task
        # self.scaler = StandardScaler()
        
        # data = np.load(data_path + "/SWaT_train.npy")

        if reduce:
            data = np.load(data_path + "/WADIA219_train_V3.npy")
        else:
            data = np.load(data_path + "/WADIA219_train.npy")
        # self.scaler.fit(data)
        # data = self.scaler.transform(data)
        test_data = np.load(data_path + '/WADIA219_test_V'+str(test_version)+'.npy')
        self.test=test_data
        # self.test = self.scaler.transform(test_data)
        
        self.train = data
        self.val = self.test
        # print(self.train.shape,self.test.shape)
        # print(np.isnan(self.train).any(),np.isnan(self.test).any())
        self.test_labels = np.load(data_path + "/WADIA219_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.task=='Forecast':
            if self.mode == "train":
                return self.train.shape[0] - self.win_size-self.step+1
            elif (self.mode == 'val'):
                return self.val.shape[0] - self.win_size-self.step+1
            elif (self.mode == 'test'):
                return self.test.shape[0] - self.win_size-self.step+1
            else:
                return self.test.shape[0] - self.win_size-self.step+1
        elif self.task=='Reconstruct':
            # print(self.train.shape)
            if self.mode == "train":
                return (self.train.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'val'):
                return (self.val.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'test'):
                return (self.test.shape[0] - self.win_size) // self.step + 1
            else:
                return (self.test.shape[0] - self.win_size) // self.win_size + 1
    def __getitem__(self, index):
        # index = index * self.step
        if self.task=='Forecast':
            if self.mode == "train":
                return np.float32(self.train[index:index + self.win_size,:].T), np.float32(self.train[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'val'):
                return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.val[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'test'):
                return np.float32(self.test[index:index + self.win_size]).T,np.float32(self.test[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'thre'):
                return np.float32(self.test[index:index + self.win_size]).T,np.float32(self.test[index+self.win_size:index+self.win_size+self.step,:]).T,np.float32(self.test_labels[index+self.win_size:index+self.win_size+self.step])
        elif self.task=='Reconstruct':
            index = index * self.step
            if self.mode == "train":
                return np.float32(self.train[index:index + self.win_size].T),np.float32(self.train[index:index + self.win_size].T)# np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'val'):
                return np.float32(self.val[index:index + self.win_size]),np.float32(self.val[index:index + self.win_size])# np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'test'):
                return np.float32(self.test[index:index + self.win_size].T),np.float32(self.test[index:index + self.win_size].T)
            # np.float32(
            #         self.test_labels[index:index + self.win_size])
            else:
                return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T,\
                       np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T,  \
                       np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
                       
class WADISegLoader(object):
    def __init__(self, data_path, win_size, step, test_version=1,mode="train",task='Forecast',reduce=False,data_scale='std'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.task=task
        if data_scale=='std':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(clip=True)
        # data = np.load(data_path + "/SWaT_train.npy")
        # print(reduce)
        if reduce:
            data = np.load(data_path + "/WADIA219_train_V3.npy")
        else:
            data = np.load(data_path + "/WADIA219_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + '/WADIA219_test_V'+str(test_version)+'.npy')
        self.test=test_data
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        self.val = self.test
        # print(self.train.shape,self.test.shape)
        # print(np.isnan(self.train).any(),np.isnan(self.test).any())
        self.test_labels = np.load(data_path + "/WADIA219_test_label.npy")
        print('reduce',reduce,test_version)
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):
        if self.task=='Forecast':
            if self.mode == "train":
                return self.train.shape[0] - self.win_size-self.step+1
            elif (self.mode == 'val'):
                return self.val.shape[0] - self.win_size-self.step+1
            elif (self.mode == 'test'):
                return self.test.shape[0] - self.win_size-self.step+1
            else:
                return self.test.shape[0] - self.win_size-self.step+1
        elif self.task=='Reconstruct':
            # print(self.train.shape)
            if self.mode == "train":
                return (self.train.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'val'):
                return (self.val.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'test'):
                return (self.test.shape[0] - self.win_size) // self.step + 1
            else:
                return (self.test.shape[0] - self.win_size) // self.win_size + 1
    def __getitem__(self, index):
        # index = index * self.step
        if self.task=='Forecast':
            if self.mode == "train":
                return np.float32(self.train[index:index + self.win_size,:].T), np.float32(self.train[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'val'):
                return np.float32(self.val[index:index + self.win_size]).T, np.float32(self.val[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'test'):
                return np.float32(self.test[index:index + self.win_size]).T,np.float32(self.test[index+self.win_size:index+self.win_size+self.step,:]).T
            elif (self.mode == 'thre'):
                return np.float32(self.test[index:index + self.win_size]).T,np.float32(self.test[index+self.win_size:index+self.win_size+self.step,:]).T,np.float32(self.test_labels[index+self.win_size:index+self.win_size+self.step])
        elif self.task=='Reconstruct':
            index = index * self.step
            if self.mode == "train":
                return np.float32(self.train[index:index + self.win_size].T),np.float32(self.train[index:index + self.win_size].T)# np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'val'):
                return np.float32(self.val[index:index + self.win_size]),np.float32(self.val[index:index + self.win_size])# np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'test'):
                return np.float32(self.test[index:index + self.win_size].T),np.float32(self.test[index:index + self.win_size].T)
            # np.float32(
            #         self.test_labels[index:index + self.win_size])
            else:
                return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T,\
                       np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]).T,  \
                       np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
                       

def get_loader_segment(data_path, batch_size, win_size=100, step=1,test_version=1, mode='train', dataset='KDD',task='Forecast',reduce=False,data_scale='std'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, test_version,mode,task=task,reduce=reduce)#,data_scale=data_scale)
    elif dataset == 'WADI':
        dataset = WADISegLoader(data_path, win_size, step, test_version,mode,task=task,reduce=reduce,data_scale=data_scale)
    elif dataset == 'WADIA219':
        dataset = WADIA219SegLoader(data_path, win_size, step, test_version,mode,task=task,reduce=reduce,data_scale=data_scale)
    # print(mode,len(dataset),step)
    shuffle = False
    # if mode == 'train':
    #     shuffle = True
    # print(shuffle)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader

# data_path='/Users/GaoRuobin/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/AD_Data_Normalized'
# train_loader = get_loader_segment(data_path, batch_size=32, win_size=100,
#                                         step=1,test_version=1,
#                                         mode='train',
#                                         dataset='SWaT',
#                                         task='Reconstruct')

# for i, (input_data, labels) in enumerate(train_loader):
#     print(input_data.shape,labels.shape)
