import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN2(nn.Module):
    def __init__(self):
        super(GNN2, self).__init__()
        self.linet = LiNet()
        
        self.classfier = Classfier(8192, 1)

    
    def forward(self, image):
        img_feat = self.linet(image)


        ret = self.classfier(img_feat)

        # print('ag:  ', ag_feat)
        # print('img:  ', img_feat)
        # print('ret:  ', ret)
        return ret

class LiNet(nn.Module):
    def __init__(self):
        super(LiNet, self).__init__()
        self.I1 = LiLayer(3, 4)
        self.I2 = LiLayer(4, 6)
        self.I3 = LiLayer(6, 8)
        self.I4 = LiLayer(8, 8)

        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, image):

        I1 = self.pool(self.I1(image))
        I2 = self.pool(self.I2(I1))
        I3 = self.pool(self.I3(I2))
        I4 = self.pool(self.I4(I3))

        ret = I4.view(I4.shape[0], -1) # [bz, 8192]

        return ret

class LiLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LiLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.RReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
            nn.RReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)+x
        x = self.conv2(x)
        return x

class Classfier(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Classfier, self).__init__()
        self.linear1 = nn.Linear(in_channel, 2048)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU(True)
        self.linear3 = nn.Linear(1024, out_channel)
        self.act = nn.Tanh()
    
    def forward(self, x):
        x = F.dropout(self.relu1(self.linear1(x)), p=0.3)
        x = F.dropout(self.relu2(self.linear2(x)), p=0.3)
        x = self.linear3(x)
        x = self.act(x)
        # print('sig: ', x)

        return x

if __name__ == '__main__':
    # net = GNNConv(64, 64, 6)
    node = np.load('../preprocess/BraTS20_Training_001_node_features_289_.npy').transpose(1, 0)[np.newaxis, ...]
    edge = np.load('../preprocess/BraTS20_Training_001_edge_index_289_.npy')[np.newaxis, ...]

    node = torch.Tensor(node)
    # print(node)
    edge = torch.Tensor(edge)

    

# 神佑、必杀、偷袭、强壮、防御、协力、吸血 -> 大剑