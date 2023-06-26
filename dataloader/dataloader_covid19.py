import torch
import glob
import nibabel as nib
import csv
import numpy as np
from random import shuffle
import random
import matplotlib.image as pltimg

import torch.utils.data as data

class ImageDataLoader(data.Dataset):
    def __init__(self, path, isTrain=True):
        self.path = path
        self.isTrain = isTrain
        if self.isTrain:
            self.files = glob.glob(self.path+'/train/*.npy')
        else:
            self.files = glob.glob(self.path+'/val2/*.npy')
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx].split('\\')[-1][:-4]
        # print(name)

        img = np.load(self.files[idx])

        img = (img-img.mean())/img.std()

        lbl = None
        if name[0] == 'n':
            lbl = -1
        elif name[0] == 'p':
            lbl = 0
        else:
            lbl = 1

        lbl = np.array(lbl)

        img = torch.Tensor(img).float()
        return img, lbl