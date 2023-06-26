import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num
    
    def forward(self, x, node):
        mask = torch.stack([node]*self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0, self.V-1).float(), 1)
        mask = (mask==0).float()
        return torch.mm(mask, x)
    


class GNNConv(nn.Module):
    def __init__(self, in_channel, out_channel, node_num):
        super(GNNConv, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.aggregation = AggrSum(node_num)
    
    def forward(self, x, edge_index):
        '''
        Input:
            x: shape is (bz, feat_dim, node_num) -> (bz, 64, 6)
            edge_index: shape is (bz, 2, edge_num) -> (bz, 2, 22)
        '''
        x = self.conv(x)
        l = edge_index[:, 0, ...].long()
        r = edge_index[:, 1, ...].long()
        degree = self.calDegree(l, x.shape[-1]).float()
        print('---   ', degree)

        degree_sqrt = degree.pow(-0.5)
        norm = degree_sqrt[l] * degree_sqrt[r] # [bz, edge_num]

        print(r.shape, x.shape, norm.shape)
        print(norm)
        target_feat = []

        for i in range(x.shape[0]):
            tmp = torch.index_select(x[i], dim=-1, index=r[i])[np.newaxis, ...]
            # print(tmp.shape)
            target_feat.append(tmp)
        target_feat = torch.concat(target_feat, dim=0) # [bz, feat_dim, edge_num] -> [1, 64, 22]
        # print(target_feat.shape)
        # target_feat = torch.index_select(x, dim=-1, index=r)
        print(norm.view(norm.shape[0], -1, 1).shape, target_feat.shape)
        # target_feat = norm.view(-1, 1) * target_feat
        aggr = self.aggregation(target_feat, l)

        return aggr

    def calDegree(self, l, node_num):
        index, deg = np.unique(l.cpu().numpy(), return_counts=True)
        # print('--- ', index, deg, node_num)
        deg_tensor = torch.zeros((node_num, ), dtype=torch.long)
        deg_tensor[index] = torch.from_numpy(deg)
        # print(deg_tensor)
        return deg_tensor.to(l.device)


class readGraph(nn.Module):
    def __init__(self, out_channel, node_num=6, feat_dim=64):
        super(readGraph, self).__init__()
        in_channel = feat_dim*node_num
        self.norm = nn.BatchNorm1d(node_num)
        self.linear1 = nn.Linear(in_channel, 512)
        self.linear2 = nn.Linear(512, out_channel)
        self.sig = nn.Sigmoid()

    def forward(self, x, edge_index):
        l = edge_index[:, 0, ...].long()
        r = edge_index[:, 1, ...].long()
        degree = self.calDegree(l, x.shape[-1]).float()
        x = x*degree
        x = self.norm(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = F.dropout(x, p=0.6)
        x = self.linear2(x)
        x = self.sig(x)
        return x

    def calDegree(self, l, node_num):
        index, deg = np.unique(l.cpu().numpy(), return_counts=True)
        # print('--- ', index, deg, node_num)
        deg_tensor = torch.zeros((node_num, ), dtype=torch.long)
        deg_tensor[index] = torch.from_numpy(deg)
        # print(deg_tensor)
        return deg_tensor.to(l.device)


class classfier(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(classfier, self).__init__()
        self.linear1 = nn.Linear(in_channel, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, out_channel)
        self.sig = nn.Sigmoid()
    
    
    def forward(self, x):
        x = F.dropout(self.linear1(x), p=0.6)
        x = F.dropout(self.linear2(x), p=0.6)
        x = self.sig(self.linear3(x))

        return x

class GraphNet(nn.Module):
    def __init__(self, out_channel):
        super(GraphNet, self).__init__()
        self.readNode = readGraph(1024)
        self.clsf = classfier(1024, out_channel)
    
    def forward(self, x, edge_index):
        x = self.readNode(x, edge_index)
        x = self.clsf(x)
        return x


if __name__ == '__main__':
    # net = GNNConv(64, 64, 6)
    node = np.load('../preprocess/BraTS20_Training_001_node_features_289_.npy').transpose(1, 0)[np.newaxis, ...]
    edge = np.load('../preprocess/BraTS20_Training_001_edge_index_289_.npy')[np.newaxis, ...]

    node = torch.Tensor(node)
    # print(node)
    edge = torch.Tensor(edge)
    net = readGraph(1024)
    x = net(node, edge)
    print(x.shape)
    
    

# 神佑、必杀、偷袭、强壮、防御、协力、吸血 -> 大剑