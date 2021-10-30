from models.model_bilstm import BiLSTM
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,LeaveOneOut
from sklearn.svm import SVC
import random
import numpy as np
from losses.focal_loss import FocalLoss
from model_util import save_model,load_model
import numpy as np
import os
import torch.nn as nn
import torch
import torch.utils.data as data
import csv
from sklearn.model_selection import KFold
from sklearn import preprocessing

def get_pos_feature(mode,name):
    feature = []
    tag = []
    with open("dataset" + os.sep + "5mc_"+mode+"_pos_"+name+".csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input = [float(x) for x in row]
            feature.append(input)
            tag.append(1.0)
    feature = np.array(feature)
    feature = feature.reshape(-1,41,4)
    tag = np.array(tag)
    #feature = torch.from_numpy(feature)
    #tag = torch.from_numpy(tag)
    return feature,tag

def get_neg_feature(mode,name):
    feature = []
    tag = []
    with open("dataset" + os.sep + "5mc_"+mode+"_neg_"+name+".csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input = [float(x) for x in row]
            feature.append(input)
            tag.append(0.0)
    feature = np.array(feature)
    feature = feature.reshape(-1, 41, 4)
    tag = np.array(tag)
    #feature = torch.from_numpy(feature)
    #tag = torch.from_numpy(tag)
    return feature, tag

class MyDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        feature, target = self.features[index], self.labels[index]
        return feature, target

    def __len__(self):
        return len(self.features)

def get_one_hot_loader(mode,features,labels):
    #features,labels = one_hot(sentences,targets)
    if(mode=='train'):
        loader = torch.utils.data.DataLoader(MyDataset(features, labels), batch_size=2048, shuffle=True)
    if (mode == 'test'):
        loader = torch.utils.data.DataLoader(MyDataset(features, labels), batch_size=2048, shuffle=False)
    return loader



seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
random.seed(seed)
np.random.seed(seed)

def train(train_loader):
    totol_loss = 0
    y_predicts, y_labels = [], []

    for data_x,data_y in train_loader:
        model.train()
        data_x = data_x.type(torch.FloatTensor).to("cuda")
        data_y = data_y.type(torch.LongTensor).to("cuda")
        pred = model(data_x)
        loss = criterion(pred, data_y)
        totol_loss = totol_loss + loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            outputs = model(data_x)
            out_max = outputs.detach()
            out_max = torch.softmax(out_max, dim=-1)
            out_max = torch.argmax(out_max, dim=1)
            out_max = out_max.to("cpu")
            y_predicts.extend(out_max)
            y_labels.extend(data_y.to("cpu").numpy())

    y_predicts = np.array(y_predicts)
    train_acc, m = model.result(y_predicts, y_labels)
    return train_acc,m

def eval(test_loader):
    model.eval()
    y_predicts = []
    y_labels = []
    for test_x,test_y in test_loader:
        with torch.no_grad():
            test_x = test_x.type(torch.FloatTensor).to("cuda")
            outputs = model(test_x)
            out_max = outputs.detach()
            out_max = torch.softmax(out_max, dim=-1)
            out_max = torch.argmax(out_max, dim=1)
            out_max = out_max.to("cpu")
            y_predicts.extend(out_max)
            y_labels.extend(test_y.numpy())
    y_predicts = np.array(y_predicts)
    dev_acc, m = model.result(y_predicts, y_labels)
    return dev_acc, m

if __name__=="__main__":
    feature_dim = [[41, 4]]
    feature_name = ["one-hot&NPC"]
    dims_1 = feature_dim[0]
    dim2_1 = dims_1[0]
    input_size = dims_1[1]
    loo = LeaveOneOut()
    #--------------------------------------------get_data----------------------------------------
    '''
    one_hot_pos_data, pos_tag = get_pos_feature("train", "one_hot")
    one_hot_neg_data, neg_tag = get_neg_feature("train", "one_hot")

    ncf_pos_data,_ = get_pos_feature("train","ncf")
    ncf_neg_data,_ = get_neg_feature("train","ncf")

    one_hot_X = np.concatenate((one_hot_pos_data,one_hot_neg_data),0)
    ncf_X = np.concatenate((ncf_pos_data,ncf_neg_data),0)

    X = np.concatenate((one_hot_X,ncf_X),-1)
    Y = np.concatenate((pos_tag,neg_tag),0)
    '''
    ncf_pos_data, pos_tag = get_pos_feature("train", "ncf")
    ncf_neg_data, neg_tag = get_neg_feature("train", "ncf")

    X = np.concatenate((ncf_pos_data, ncf_neg_data), 0)
    Y = np.concatenate((pos_tag, neg_tag), 0)

    kflod = KFold(n_splits=5, shuffle=True, random_state=0)
    kflod1 = KFold(n_splits=11, shuffle=True, random_state=0)
    j = 0
    for train_index,test_index in kflod.split(X):
        print("------------------------------------------------------")
        train_x = X[train_index]
        train_y = Y[train_index]
        test_x = X[test_index]
        test_y = Y[test_index]
        test_loader = get_one_hot_loader("test",test_x,test_y)

        pos_data = []
        neg_data = []
        pos_tag = []
        neg_tag = []
        for i in range(len(train_y)):
            if(train_y[i]==1):
                pos_data.append(train_x[i])
                pos_tag.append(train_y[i])
            else:
                neg_data.append(train_x[i])
                neg_tag.append(train_y[i])
        neg_data = np.array(neg_data)
        neg_tag = np.array(neg_tag)
        pos_data = np.array(pos_data)
        pos_tag = np.array(pos_tag)
        k = 0
        for train_index, test_index in kflod1.split(neg_data):
            print("k=",k)
            neg_x = neg_data[test_index]
            neg_y = neg_tag[test_index]
            sentences = np.concatenate((pos_data, neg_x), 0)
            tags = np.concatenate((pos_tag, neg_y), 0)
            train_loader = get_one_hot_loader("train", sentences, tags)
            model = BiLSTM(input_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            # criterion = FocalLoss()
            model = model.to("cuda")
            trainacc = 0
            trainm = np.zeros((2, 2))
            dev_acc = 0
            dev_m = np.zeros((2, 2))
            for epoch in range(200):
                train_acc, train_m = train(train_loader)
                acc, m = eval(test_loader)
                if train_acc > trainacc:
                    dev_acc = acc
                    dev_m = m
                    trainacc = train_acc
                    trainm = train_m
                    save_model(model,"ncf_train_"+str(j)+os.sep, str(k))
            print("dev_acc", dev_acc)
            print("dev_m:", dev_m)
            k = k + 1
        j = j+1


