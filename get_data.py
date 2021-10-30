import csv
import numpy as np
import os
import torch.nn as nn
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing


class MyDataset(data.Dataset):
    def __init__(self, feature1,feature2,feature3,label):
        self.feature1 = feature1
        self.feature2 = feature2
        self.feature3 = feature3
        self.label = label

    def __getitem__(self, index):#返回的是tensor
        feature1,feature2,feature3,target = self.feature1[index],self.feature2[index], self.feature3[index],self.label[index]
        return feature1,feature2,feature3,target

    def __len__(self):
        return len(self.feature1)

def get_loader(mode,feature1,feature2,feature3,label):
    if(mode=='train'):
        loader = torch.utils.data.DataLoader(MyDataset(feature1,feature2,feature3,label), batch_size=2048, shuffle=True)
    if (mode == 'test'):
        loader = torch.utils.data.DataLoader(MyDataset(feature1,feature2,feature3,label), batch_size=2048, shuffle=False)
    return loader


def get_pos_feature(mode,name,dim1,dim2):
    feature = []
    tag = []
    with open("./features" + os.sep + "5mc_"+mode+"_pos_"+name+".csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input = [float(x) for x in row]
            feature.append(input)
            tag.append(1.0)
    feature = np.array(feature)
    feature = feature.reshape(-1,dim1,dim2)
    tag = np.array(tag)
    feature = torch.from_numpy(feature)
    tag = torch.from_numpy(tag)
    return feature,tag

def get_neg_feature(mode,name,dim1,dim2):
    feature = []
    tag = []
    with open("./features" + os.sep + "5mc_"+mode+"_neg_"+name+".csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input = [float(x) for x in row]
            feature.append(input)
            tag.append(0.0)
    feature = np.array(feature)
    feature = feature.reshape(-1, dim1, dim2)
    tag = np.array(tag)
    feature = torch.from_numpy(feature)
    tag = torch.from_numpy(tag)
    return feature, tag
