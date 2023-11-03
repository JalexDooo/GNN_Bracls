import torch
import glob
import nibabel as nib
import csv
import numpy as np
from random import shuffle
import random

import torch.utils.data as data

class ImageLoader(data.DataLoader):
    def __init__(self, csv_path, seg_path, isTrain=True, segmentation=None, all=False):
        self.segmentation = segmentation
        if segmentation is None:
            self.seg_path = seg_path
            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                csvData = csv.reader(f)
                for i in csvData:
                    data.append(i)
            
            data = np.array(data)
            
            # clean data remove 'BraTS20_Training_084', because case84 is alive.
            idx = (data[:, 3] == 'GTR') * (data[:, 0] != 'BraTS20_Training_084')
            tmp = data[idx]
            self.data = tmp
            length = len(self.data)
            # shuffle(self.data)
            if isTrain:
                self.data = self.data[:int(0.7*length)]
            else:
                self.data = self.data[int(0.7*length):]
            if all:
                self.data = tmp[:]

        else:
            self.seg_path = seg_path
            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                csvData = csv.reader(f)
                for i in csvData:
                    data.append(i)
            data = np.array(data)

            
            # clean data remove 'BraTS20_Training_084'
            idx = (data[:, 2] == 'GTR')
            tmp = data[idx]
            self.data = tmp
            length = len(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.segmentation is None:
            age = float(self.data[idx][1])
            age = (age-54)/36

            name = self.data[idx][0]

            survival = float(self.data[idx][2])
            # survival = (survival-450)/350
            survival = survival/(365*5)

            image_path = self.seg_path+'/'+self.data[idx][0]+'/'+self.data[idx][0]+'_t1ce.nii.gz'
            image_arr = nib.load(image_path).get_fdata()

            label_path = self.seg_path+'/'+self.data[idx][0]+'/'+self.data[idx][0]+'_seg.nii.gz'
            label_arr = nib.load(label_path).get_fdata()
            image_arr, label_arr = self.process(image_arr, label_arr)
            image_arr = (image_arr - image_arr.mean())/image_arr.std()
            image_arr = torch.from_numpy(image_arr).float()
            label_arr = torch.from_numpy(label_arr).float()

            return name, image_arr, label_arr, age, survival
        else:
            age = float(self.data[idx][1])
            age = (age-54)/36

            image_path = self.seg_path+'/'+self.data[idx][0]+'/'+self.data[idx][0]+'_t1ce.nii.gz'
            image_arr = nib.load(image_path).get_fdata()

            label_path = self.segmentation+'/'+self.data[idx][0]+'.nii.gz'
            label_arr = nib.load(label_path).get_fdata()
            image_arr, label_arr = self.process(image_arr, label_arr)
            image_arr = (image_arr - image_arr.mean())/image_arr.std()
            image_arr = torch.from_numpy(image_arr).float()
            label_arr = torch.from_numpy(label_arr).float()

            name = self.data[idx][0]

            return name, image_arr, label_arr, age
    
    def process(self, image, label):
        '''
        Got nonzero center matrix with shape of (128, 128, 64)
        '''
        mask = label>0
        index = np.nonzero(mask)
        x = (index[0][0]+index[0][-1])//2 + random.randint(-10, 10)
        y = (index[1][0]+index[1][-1])//2 + random.randint(-10, 10)
        z = (index[2][0]+index[2][-1])//2 + random.randint(-10, 10)
        if x-64<0:
            xx1 = 0
            xx2 = 128
        elif x+64>240:
            xx1 = 240-128
            xx2 = 240
        else:
            xx1 = x-64
            xx2 = x+64
        if y-64<0:
            yy1 = 0
            yy2 = 128
        elif y+64>240:
            yy1 = 240-128
            yy2 = 240
        else:
            yy1 = y-64
            yy2 = y+64
        
        if z-32<0:
            zz1 = 0
            zz2 = 64
        elif z+32>240:
            zz1 = 240-64
            zz2 = 240
        else:
            zz1 = z-32
            zz2 = z+32
        
        image = image[xx1:xx2, yy1:yy2, zz1:zz2]
        label = label[xx1:xx2, yy1:yy2, zz1:zz2]
        return image, label


class ImageNodeLoader(data.DataLoader):
    def __init__(self, csv_path, seg_path, node_npy, edge_npy, isTrain=True, segmentation=None, all=False):
        self.segmentation = segmentation
        self.node_path = node_npy # node_npy's root path
        self.edge_path = edge_npy
        if segmentation is None:
            self.seg_path = seg_path
            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                csvData = csv.reader(f)
                for i in csvData:
                    data.append(i)
            
            data = np.array(data)
            
            # clean data remove 'BraTS20_Training_084'
            idx = (data[:, 3] == 'GTR') * (data[:, 0] != 'BraTS20_Training_084')
            tmp = data[idx]
            self.data = tmp
            length = len(self.data)
            # shuffle(self.data)
            if isTrain:
                self.data = self.data[:int(0.7*length)]
            else:
                self.data = self.data[int(0.7*length):]
            if all:
                self.data = tmp[:]

        else:
            self.seg_path = seg_path
            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                csvData = csv.reader(f)
                for i in csvData:
                    data.append(i)
            data = np.array(data)

            
            # clean data remove 'BraTS20_Training_084'
            idx = (data[:, 2] == 'GTR')
            tmp = data[idx]
            self.data = tmp
            length = len(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.segmentation is None:
            age = float(self.data[idx][1])
            age = (age-54)/36

            name = self.data[idx][0]

            survival = float(self.data[idx][2])
            # survival = (survival-450)/350
            survival = survival/(365*5)

            # load node and edge features.
            node_feat = torch.from_numpy(np.load(self.node_path+'/'+name+'_node_features_{}_.npy'.format(survival))).float()
            edge_feat = torch.from_numpy(np.load(self.edge_path+'/'+name+'_edge_features_{}_.npy'.format(survival))).float()

            image_path = self.seg_path+'/'+self.data[idx][0]+'/'+self.data[idx][0]+'_t1ce.nii.gz'
            image_arr = nib.load(image_path).get_fdata()

            label_path = self.seg_path+'/'+self.data[idx][0]+'/'+self.data[idx][0]+'_seg.nii.gz'
            label_arr = nib.load(label_path).get_fdata()
            image_arr, label_arr = self.process(image_arr, label_arr)
            image_arr = (image_arr - image_arr.mean())/image_arr.std()
            image_arr = torch.from_numpy(image_arr).float()
            label_arr = torch.from_numpy(label_arr).float()

            return name, image_arr, label_arr, age, survival, node_feat, edge_feat
        else:
            age = float(self.data[idx][1])
            age = (age-54)/36

            image_path = self.seg_path+'/'+self.data[idx][0]+'/'+self.data[idx][0]+'_t1ce.nii.gz'
            image_arr = nib.load(image_path).get_fdata()

            label_path = self.segmentation+'/'+self.data[idx][0]+'.nii.gz'
            label_arr = nib.load(label_path).get_fdata()
            image_arr, label_arr = self.process(image_arr, label_arr)
            image_arr = (image_arr - image_arr.mean())/image_arr.std()
            image_arr = torch.from_numpy(image_arr).float()
            label_arr = torch.from_numpy(label_arr).float()

            name = self.data[idx][0]

            return name, image_arr, label_arr, age, node_feat, edge_feat
    
    def process(self, image, label):
        '''
        Got nonzero center matrix with shape of (128, 128, 64)
        '''
        mask = label>0
        index = np.nonzero(mask)
        x = (index[0][0]+index[0][-1])//2 + random.randint(-10, 10)
        y = (index[1][0]+index[1][-1])//2 + random.randint(-10, 10)
        z = (index[2][0]+index[2][-1])//2 + random.randint(-10, 10)
        if x-64<0:
            xx1 = 0
            xx2 = 128
        elif x+64>240:
            xx1 = 240-128
            xx2 = 240
        else:
            xx1 = x-64
            xx2 = x+64
        if y-64<0:
            yy1 = 0
            yy2 = 128
        elif y+64>240:
            yy1 = 240-128
            yy2 = 240
        else:
            yy1 = y-64
            yy2 = y+64
        
        if z-32<0:
            zz1 = 0
            zz2 = 64
        elif z+32>240:
            zz1 = 240-64
            zz2 = 240
        else:
            zz1 = z-32
            zz2 = z+32
        
        image = image[xx1:xx2, yy1:yy2, zz1:zz2]
        label = label[xx1:xx2, yy1:yy2, zz1:zz2]
        return image, label



class DataLoader(data.Dataset):
    def __init__(self, path, isTrain=True):
        self.path = path
        self.node = glob.glob(self.path+'/*node*.npy')
        self.edge = glob.glob(self.path+'/*edge*.npy')
        length = len(self.node)
        idx = list(range(length))
        shuffle(idx)
        self.rand_node = []
        self.rand_edge = []
        for i in idx:
            self.rand_node.append(self.node[i])
            self.rand_edge.append(self.edge[i])
        if isTrain:
            self.node = self.rand_node[:int(0.7*length)]
            self.edge = self.rand_edge[:int(0.7*length)]
        else:
            self.node = self.rand_node[int(0.7*length):]
            self.edge = self.rand_edge[int(0.7*length):]
    
    def __len__(self):
        return len(self.node)

    def __getitem__(self, idx):
        node = self.node[idx]
        edge = self.edge[idx]
        label = np.int64(node.split('_')[-2])
        label = label/(365*5)

        node_feat = torch.from_numpy(np.load(node)).float()
        edge_index = torch.from_numpy(np.load(edge)).float()
        # label = torch.Tensor(label)
        

        return node_feat, edge_index, label