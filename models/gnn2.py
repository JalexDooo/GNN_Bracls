import numpy as np
import torch
import math
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

class GNN1(nn.Module):
    def __init__(self):
        super(GNN1, self).__init__()
        self.linet = LiNet()
        self.agnet = AgNet()
        
        self.classfier = Classfier(4096, 1)

    
    def forward(self, image, label, age):
        img_feat = self.linet(image, label)
        ag_feat = self.agnet(age)
        feat = img_feat+ag_feat
        ret = self.classfier(feat)

        # print('ag:  ', ag_feat)
        # print('img:  ', img_feat)
        # print('ret:  ', ret)
        return ret

class AgNet(nn.Module):
    def __init__(self):
        super(AgNet, self).__init__()
        self.l1 = nn.Linear(1, 1024)
        self.l2 = nn.Linear(1024, 4096)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        l1 = self.act1(self.l1(x))
        l1 = F.dropout(l1, p=0.4)
        l2 = self.l2(l1)
        # l2 = F.dropout(l2, p=0.4)

        # print('ag:  ', l2)

        return l2


class LiNet(nn.Module):
    def __init__(self):
        super(LiNet, self).__init__()
        self.I1 = LiLayer(1, 2)
        self.L1 = LiLayer(1, 2)
        self.I2 = LiLayer(2, 4)
        self.L2 = LiLayer(2, 4)
        self.I3 = LiLayer(4, 8)
        self.L3 = LiLayer(4, 8)
        self.I4 = LiLayer(8, 24)
        self.L4 = LiLayer(8, 24)
        self.pool = nn.MaxPool3d(2, 2)


    def forward(self, image, label):

        I1 = self.pool(self.I1(image))
        L1 = self.pool(self.L1(label))
        I2 = self.pool(self.I2(I1+L1))
        L2 = self.pool(self.L2(L1))
        I3 = self.pool(self.I3(I2+L2))
        L3 = self.pool(self.L3(L2))
        I4 = self.pool(self.I4(I3+L3))
        L4 = self.pool(self.L4(L3))
        ret = I4+L4
        ret = ret.view(I4.shape[0], -1) # [bz, 4096]

        # print('LI:  ', ret)

        return ret
    
class Attention(nn.Module): # hidden_size:256    6144/256=24
    def __init__(self, deep):
        super(Attention, self).__init__()
        self.num_att_heads = 24
        self.att_head_size = int(deep / self.num_att_heads) # 256 / 16 = 16
        self.all_head_size = self.num_att_heads * self.att_head_size # 16 * 16 = 256

        self.query = nn.Linear(deep, self.all_head_size)
        self.key = nn.Linear(deep, self.all_head_size)
        self.value = nn.Linear(deep, self.all_head_size)

        self.out = nn.Linear(deep, deep)

        self.softmax = nn.Softmax(dim=-1)
    
    def reshape_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_att_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, -1)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # print(q.shape, k.shape, v.shape)

        # q = self.reshape_for_scores(q)
        # k = self.reshape_for_scores(k)
        # v = self.reshape_for_scores(v)


        qk = torch.matmul(q, k.transpose(-1, -2))
        qk = qk / math.sqrt(self.att_head_size)
        qk = self.softmax(qk)

        qkv = torch.matmul(qk, v)
        qkv = qkv.permute(0, 2, 1).contiguous()
        new_qkv_shape = qkv.size()[:-2] + (self.all_head_size,)
        qkv = qkv.view(*new_qkv_shape)

        att_output = self.out(qkv)

        # print('attout: ', att_output.shape)

        return att_output


class LiLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LiLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm3d(in_channel),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.RReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(in_channel),
            nn.Conv3d(in_channel, out_channel, kernel_size=1, padding=0),
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
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        x = F.dropout(self.relu1(self.linear1(x)), p=0.3)
        x = F.dropout(self.relu2(self.linear2(x)), p=0.3)
        x = self.linear3(x)
        x = self.act(x)
        # print('sig: ', x)

        return x

class GraphNet(nn.Module):
    def __init__(self, out_channel):
        super(GraphNet, self).__init__()
        self.readG1 = readGraph(6144)
        self.readG2 = readGraph(6144, 3, 64)
        self.att1 = Attention(6144)
        self.att2 = Attention(6144)

        self.img_encoding = LiNet()

        self.clsf = Classfier(6144, out_channel)
    
    def forward(self, img_arr, seg_arr, age, node1, edge1, node2, edge2): # age is including node & edge
        '''
        img_arr: [bz, 1, 128, 128, 64]
        seg_arr: [bz, 1, 128, 128, 64]
        age : not used and including in node & edge
        node1: [4, 6, 64]
        edge1: [4, 2, 22]
        node2: [4, 3, 64]
        edge2: [4, 2, 9]
        '''
        img_encoding = self.img_encoding(img_arr, seg_arr).view(seg_arr.size(0), -1) # [bz, 6144]
        g1 = self.readG1(node1, edge1)
        g2 = self.readG2(node2, edge2)
        x = self.att1(g1)+self.att2(g2)
        # print('g1, g2: ', g1.shape, g2.shape)
        print(x.shape)
        x = self.clsf(g1+g2+img_encoding)
        return x


if __name__ == '__main__':
    # # net = GNNConv(64, 64, 6)
    # node = np.load('../preprocess/BraTS20_Training_001_node_features_289_.npy').transpose(1, 0)[np.newaxis, ...]
    # edge = np.load('../preprocess/BraTS20_Training_001_edge_index_289_.npy')[np.newaxis, ...]

    # node = torch.Tensor(node)
    # # print(node)
    # edge = torch.Tensor(edge)
    # net = readGraph(1024)
    # x = net(node, edge)
    # print(x.shape)

