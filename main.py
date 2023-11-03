import csv
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt

import torch
import dataloader
import models
from torch.utils.data import DataLoader
import torch.optim as optim


# D:\Study\dataset\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData
# D:\Study\sub1

# /sunjindong/dataset/MICCAI_BraTS2020_TrainingData
# /sunjindong/dataset/survival_info.csv

# csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'


def train(**kwargs):
    csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'
    seg_path = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    # csvPath = '/sunjindong/dataset/survival_info.csv'
    # seg_path = '/sunjindong/dataset/MICCAI_BraTS2020_TrainingData'
    data = dataloader.ImageLoader(csvPath, seg_path, all=True)
    datas = DataLoader(data, batch_size=16, shuffle=True, num_workers=0)

    model = models.GNN1()
    criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(94):
        losses = []
        for _, image, label, age, survival in datas:
            # print(image.shape, label.shape, age, survival)



            optimizer.zero_grad()
            image = image[:, np.newaxis].float()
            label = label[:, np.newaxis].float()
            survival = survival[:, np.newaxis].float()
            age = torch.unsqueeze(age, dim=1).float()
            x = model(image, label, age)
            # print('ourput: ', x.shape, survival.shape, age)
            # print('out:  ', x)
            # print('survival:  ', survival)


            loss = criterion(x, survival)

            loss.backward()
            optimizer.step()
            losses += [float(loss)]
            # break
        losses = np.array(losses)
        
        print('epoch: {}, loss: {}'.format(epoch, losses.mean()))
        # break
    torch.save(model.state_dict(), './ckpt_g1_94.pth')
    
    data = dataloader.ImageLoader(csvPath, seg_path, isTrain=False)
    datas = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    # model.load_state_dict(torch.load('./ckpt.pth'))
    model.eval()

    
    for _, image, label, age, survival in datas:
        image = image[:, np.newaxis].float()
        label = label[:, np.newaxis].float()
        survival = survival[:, np.newaxis].float()
        age = torch.unsqueeze(age, dim=1).float()
        x = model(image, label, age)

        print('Survival: {}, Predict: {}'.format(survival*365*5, x*365*5))


def test(**kwargs):
    csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/survival_evaluation.csv'
    imgPath = 'D:/Study/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    segPath = 'D:/Study/submission'

    data = dataloader.ImageLoader(csvPath, imgPath, True, segPath)
    datas = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    model = models.GNN1()
    model.load_state_dict(torch.load('./ckpt110.pth'))

    model.eval()

    printCSV = []

    for name, image, label, age in datas:
        image = image[:, np.newaxis].float()
        label = label[:, np.newaxis].float()


        age = torch.unsqueeze(age, dim=1).float()
        x = model(image, label, age)

        x = float(x)*365*5
        printCSV.append([name[0], str(x)])

    
    with open('./submision110.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # for i in printCSV:
        #     print(i)
        # writer.writerow(['BrasTS20ID','Survival'])
        writer.writerows(printCSV)



def draw(**kwargs):
    csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/survival_evaluation.csv'
    imgPath = 'D:/Study/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    segPath = 'D:/Study/submission'

    img = 'D:/Study/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_001/BraTS20_Validation_001_t1ce.nii.gz'
    lbl = 'D:/Study/submission/BraTS20_Validation_001.nii.gz'
    img_arr = nib.load(img)
    img_arr = img_arr.get_fdata()
    lbl_arr = nib.load(lbl)
    lbl_arr = lbl_arr.get_fdata()
    print(img_arr.shape)
    print(lbl_arr.shape)

    # plt.axis('off')
    # plt.xticks([])
    # plt.imshow(img_arr[:, :, 100], cmap='gray')
    # plt.savefig('./img.jpg', bbox_inches='tight', pad_inches=-0.1)
    # plt.show()

    # plt.axis('off')
    # plt.xticks([])
    # plt.imshow(lbl_arr[:, :, 100], cmap='inferno')
    # plt.savefig('./lbl.jpg', bbox_inches='tight', pad_inches=-0.1)
    # plt.show()

    # plt.savefig()

    data = dataloader.ImageLoader(csvPath, imgPath, True, segPath)
    datas = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    for name, image, label, age in datas:
        print(name)
        print(image.shape)
        print(label.shape)
        print(age)

        plt.axis('off')
        plt.xticks([])
        plt.imshow(image[0, :, :, 38], cmap='gray')
        plt.savefig('./img1.jpg', bbox_inches='tight', pad_inches=-0.1)
        plt.show()

        plt.axis('off')
        plt.xticks([])
        plt.imshow(label[0, :, :, 38], cmap='inferno')
        plt.savefig('./lbl1.jpg', bbox_inches='tight', pad_inches=-0.1)
        plt.show()


        break


def train_val(**kwargs):
    csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'
    seg_path = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    # csvPath = '/sunjindong/dataset/survival_info.csv'
    # seg_path = '/sunjindong/dataset/MICCAI_BraTS2020_TrainingData'

    model = models.GNN1()
    
    data = dataloader.ImageLoader(csvPath, seg_path, isTrain=True, all=True)
    datas = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    model.load_state_dict(torch.load('./ckpt_g1_94.pth'))
    model.eval()
    printCSV = []

    for name, image, label, age, survival in datas:
        image = image[:, np.newaxis].float()
        label = label[:, np.newaxis].float()
        survival = survival[:, np.newaxis].float()
        age = torch.unsqueeze(age, dim=1).float()
        x = model(image, label, age)
        x = float(x)*365*5

        print('Survival: {}, Predict: {}'.format(survival*365*5, x))
        printCSV.append([name[0], str(x)])

    
    with open('./submision_tr_g1_94.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # for i in printCSV:
        #     print(i)
        # writer.writerow(['BrasTS20ID','Survival'])
        writer.writerows(printCSV)


def trainGraph(**kwargs):
    path = './preprocess'
    csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'
    seg_path = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    node_npy = './preprocess/'
    edge_npy = './preprocess/'
    data = dataloader.ImageNodeLoader(csvPath, seg_path, node_npy, edge_npy, all=True)
    datas = DataLoader(data, batch_size=16, shuffle=True, num_workers=0)
    model = models.GraphNet(1)
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(1000):
        losses = []
        for (_, image_arr, label_arr, age, label, node_feat, edge_feat) in datas:
            # print(node_feat.shape, edge_index.shape, label)
            optimizer.zero_grad()
            label = label[:, np.newaxis].float()
            x = model(image_arr, label_arr, age, node_feat, edge_feat)
            loss = criterion(x, label)
            loss.backward()
            optimizer.step()
            losses += [float(loss)]
            # print(x.shape, label.shape, loss)
        # print(losses)
        losses = np.array(losses)
        print('epoch: {}, loss: {}'.format(epoch, losses.mean()))
    
    data = dataloader.DataLoader(path, False)
    datas = DataLoader(data, batch_size=1, num_workers=0)
    model.eval()
    for (node_feat, edge_index, label) in datas:
        # print(node_feat.shape, edge_index.shape, label)
        label = label[:, np.newaxis].float()
        x = model(node_feat, edge_index)

        output = x * 365*5
        label = label * 365*5
        print('Age: {}, predict: {}'.format(label, output))


def ROC():
    # import sklearn.metrics as metrics
    import scikitplot as skplt
    from sklearn import datasets, svm, metrics
    from sklearn.model_selection import train_test_split
    
    gt = [0,2,2,2,2,0,2,2,2,2,2,1,0,1,1,2,0,2,1,1,1,0,2,2,0,0,0,2,2,2,2,0,0,1,0,1,2,2,1,0,2,2,1,0,0,1,2,0,1,0,0,0,1,0,1,1,2,0,0,0,2,1,0,0,2,0,0,0,2,0,2,0,1,1,1,1,0,2,2,0,2,1,0,2,1,2,1,1,2,2,2,1,0,2,0,0,0,2,1,2,2,2,0,2,0,2,0,2,1,0,0,0,2,2,0,2,1,0]

    base = [0,2,2,2,1,2,2,2,2,2,2,1,1,2,2,2,1,2,1,1,0,0,2,2,1,0,2,2,2,2,2,0,1,1,2,2,1,2,0,0,1,2,2,1,0,1,2,0,0,2,0,2,2,2,2,2,2,0,2,0,2,2,1,0,2,2,0,1,1,0,2,0,2,1,2,1,0,2,1,0,2,1,2,2,0,1,2,0,2,2,2,2,0,2,1,0,1,1,2,2,2,2,2,2,0,2,0,2,1,1,1,2,2,2,0,2,2,0]
    base_g1 = [0,2,1,2,1,0,1,2,2,2,2,1,0,1,1,1,1,2,1,0,1,0,1,2,0,0,1,2,2,1,2,0,0,1,0,1,1,2,0,0,2,2,0,1,0,0,2,0,1,0,0,0,0,0,2,1,1,0,1,0,2,1,0,0,2,0,0,0,1,0,1,0,1,0,1,1,0,1,1,0,2,0,0,0,2,0,0,0,0,1,2,0,0,1,0,2,0,2,1,2,1,2,0,1,2,2,0,1,0,1,2,2,1,2,1,1,2,2]
    base_g12 = [1,1,1,2,0,0,1,2,2,0,2,0,0,1,0,1,0,2,1,0,2,0,1,2,0,0,0,0,2,0,1,0,0,0,0,2,1,2,1,0,1,2,0,0,0,1,2,0,0,0,0,0,0,1,2,1,2,1,0,0,2,1,0,0,2,1,0,0,2,0,2,0,1,0,1,0,0,2,2,0,2,1,0,1,0,2,0,0,2,2,2,1,0,1,0,0,0,2,0,2,1,2,0,0,0,2,0,2,0,0,0,0,1,2,0,1,1,0]
    base_g1_age = [0,2,2,2,2,2,2,2,2,1,2,1,0,0,0,2,1,2,1,0,0,0,1,2,0,0,0,2,2,0,2,0,0,0,0,2,1,2,1,0,2,2,2,1,0,2,2,0,0,0,0,0,2,0,2,1,2,0,1,0,1,2,0,0,2,0,0,0,2,0,1,0,1,0,2,1,0,1,2,0,2,1,0,2,1,2,0,0,1,2,2,1,0,2,0,0,0,0,1,1,0,2,0,1,0,2,0,2,1,0,0,1,2,2,0,2,1,0]
    base_g12_age = [0,2,2,2,2,0,1,2,2,1,2,0,0,2,1,2,1,2,1,1,1,0,1,2,0,0,0,2,2,2,2,0,0,1,0,2,1,2,0,0,2,2,1,0,0,1,2,0,0,0,0,0,1,0,1,1,0,0,1,0,2,1,0,0,2,0,0,0,2,0,2,0,1,0,1,1,0,2,1,0,2,1,0,2,0,2,0,1,1,2,2,1,0,2,0,0,0,2,1,2,2,2,0,2,0,2,0,2,1,0,0,1,0,2,0,2,1,0]

    gt = np.array(gt)
    base = np.array(base)
    base_g1 = np.array(base_g1)
    base_g12 = np.array(base_g12)
    base_g1_age = np.array(base_g1_age)
    base_g12_age = np.array(base_g12_age)

    gt = (gt==0).astype(np.int64)
    base = (base==0).astype(np.int64)
    base_g1 = (base_g1==0).astype(np.int64)
    base_g12 = (base_g12==0).astype(np.int64)
    base_g1_age = (base_g1_age==0).astype(np.int64)
    base_g12_age = (base_g12_age==0).astype(np.int64)





    # skplt.metrics.plot_roc_curve(gt, base)
    # plt.show()




def outCSV():
    csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'
    seg_path = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    # csvPath = '/sunjindong/dataset/survival_info.csv'
    # seg_path = '/sunjindong/dataset/MICCAI_BraTS2020_TrainingData'
    data = dataloader.ImageLoader(csvPath, seg_path, all=True)
    datas = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    printCSV = []
    printCSV.append(['id', 'age', 'Survival', 'LWT', 'LET', 'LTC'])

    for name, image, label, age, survival in datas:
        # print(name[0])
        # print(image.shape, label.shape, age[0], survival[0])
        name = name[0]
        age = float(age[0]*36+54)
        survival = survival[0]*365*5
        if survival < 10*30:
            cls = 'survival<10'
        elif survival < 15*30 and survival >= 10*30:
            cls = '10<survival<15'
        else:
            cls = 'survival>15'
        # III = int((image>0).sum())
        LWT = int((label==2).sum()) #/III
        LET = int((label==1).sum()) #/III
        LTC = int((label==4).sum()) #/III
        printCSV.append([name, str(age), cls, str(LWT), str(LET), str(LTC)])
        # break

    with open('./Bratsdata2.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # for i in printCSV:
        #     print(i)
        # writer.writerow(['BrasTS20ID','Survival'])
        writer.writerows(printCSV)


def drawplot():
    import seaborn as sns
    import pandas as pd
    #Import datset
    data = pd.read_csv('./Bratsdata2.csv')
    # print(data.head(10))
    # print(data.info())

    # data.drop(['Unnamed: 32','id'], axis = 1 , inplace=True)

    # print(data.describe())

    # print(data.skew())

    print(data.columns[0])

    print(data)

    #Visualizing Multidimensional Relationships
    plt.style.use('fivethirtyeight')
    sns.set_style("white")
    sns.pairplot(data[[data.columns[1], data.columns[2],data.columns[3],data.columns[4],
                        data.columns[5]]], hue='Survival')

    # g = sns.pairplot(data[[data.columns[0], data.columns[1],data.columns[2],data.columns[3],
    #                      data.columns[4], data.columns[5]]], hue = 'diagnosis' ,)
    # g.map_lower(sns.kdeplot, levels=3)
    plt.savefig('./pairplot3.jpg')
    plt.show()


# def test():
#     tmp = np.load('./preprocess/BraTS20_Training_001_node_features.npy')
#     print(tmp)
#     pass



def weight_image(image, label):
    step = image.shape[0]//8
    image_feat = []
    label_feat = []
    et_feat = []
    tc_feat = []
    wt_feat = []

    # image_feat
    for i in range(8):
        for j in range(8):
            feat = (image[i*step:i*step+step, j*step:j*step+step, :]>0).sum()
            image_feat.append(feat/(30*30*155))

    # label_feat
    for i in range(8):
        for j in range(8):
            feat = (label[i*step:i*step+step, j*step:j*step+step, :]>0).sum()
            label_feat.append(feat/(30*30*155))

    # et_feat
    for i in range(8):
        for j in range(8):
            feat = (label[i*step:i*step+step, j*step:j*step+step, :]==4).sum()
            et_feat.append(feat/(30*30*155))

    # tc_feat
    for i in range(8):
        for j in range(8):
            feat = (label[i*step:i*step+step, j*step:j*step+step, :]==1).sum()
            tc_feat.append(feat/(30*30*155))

    # wt_feat
    for i in range(8):
        for j in range(8):
            feat = (label[i*step:i*step+step, j*step:j*step+step, :]==2).sum()
            wt_feat.append(feat/(30*30*155))

    return image_feat, label_feat, et_feat, tc_feat, wt_feat

def edge():
    edge_index = []
    edge_index.append([0, 1])
    edge_index.append([0, 2])
    edge_index.append([0, 3])
    edge_index.append([1, 2])
    edge_index.append([2, 3])
    edge_index.append([3, 4])
    edge_index.append([3, 5])
    edge_index.append([4, 5])

    self_index = []
    for i in range(6):
        self_index.append([i, i])
    self_index = np.array(self_index)
    self_index = np.array(self_index).transpose(1, 0)

    edge_index = np.array(edge_index).transpose(1, 0)
    edge_index = np.concatenate((edge_index, edge_index[::-1, :]), axis=1)
    edge_index = np.concatenate((edge_index, self_index), axis=1)

    return edge_index

def edge2():
    edge_index = []
    edge_index.append([0, 1])
    edge_index.append([0, 2])
    edge_index.append([1, 2])

    self_index = []
    for i in range(6):
        self_index.append([i, i])
    self_index = np.array(self_index)
    self_index = np.array(self_index).transpose(1, 0)

    edge_index = np.array(edge_index).transpose(1, 0)
    edge_index = np.concatenate((edge_index, edge_index[::-1, :]), axis=1)
    edge_index = np.concatenate((edge_index, self_index), axis=1)

    return edge_index

def process_data():
    csvPath = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'
    segPath = 'D:/Study/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    data = []
    npdata = []
    with open(csvPath, 'r', encoding='utf-8') as f:
        csvData = csv.reader(f)
        for i in csvData:
            data.append(i)
    
    data = np.array(data)
    
    # clean data remove 'BraTS20_Training_084'
    idx = (data[:, 3] == 'GTR') * (data[:, 0] != 'BraTS20_Training_084')
    tmp = data[idx]
    data = tmp

    for name, age, survival, _ in data:
        imgArr = nib.load(segPath+'/'+name+'/'+name+'_t1ce.nii.gz')
        imgAff = imgArr.affine
        imgData = imgArr.get_fdata()

        lblArr = nib.load(segPath+'/'+name+'/'+name+'_seg.nii.gz')
        lblData = lblArr.get_fdata()

        # age : (x-18)/72 (x-54)/36
        age = np.array([(float(age)-18)/72]*64)
        image_feat, label_feat, et_feat, tc_feat, wt_feat = weight_image(imgData, lblData)

        node_features = np.array([age, image_feat, label_feat, et_feat, tc_feat, wt_feat])
        node_features2 = np.array([et_feat, tc_feat, wt_feat])
        print(node_features.shape)

        edge_index = edge()
        edge_index2 = edge2()
        print(edge_index.shape)
        print(edge_index)

        np.save('./preprocess/'+name+'_node_features_{}_.npy'.format(survival), node_features)
        np.save('./preprocess/'+name+'_edge_index_{}_.npy'.format(survival), edge_index)
        np.save('./preprocess/'+name+'_node_features2_{}_.npy'.format(survival), node_features2)
        np.save('./preprocess/'+name+'_edge_index2_{}_.npy'.format(survival), edge_index2)

def model_intput_shape():
    node1 = torch.randn((4, 6, 64))
    edge1 = torch.randn((4, 2, 22))

    node2 = torch.randn((4, 3, 64))
    edge2 = torch.randn((4, 2, 9))

    img = torch.randn((4, 1, 128, 128, 64))
    seg = torch.randn((4, 1, 128, 128, 64))

    model = models.GraphNet(1)

    age = None # age is encoding in node1 and edge1
    out = model(img, seg, age, node1, edge1, node2, edge2)
    print(out.shape)


if __name__ == '__main__':
    import fire
    fire.Fire()
