from get_RNA_data import read_file
from models.model_bilstm import BiLSTM
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,LeaveOneOut
import random
import numpy as np
import torch.nn.functional as F

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)

def train():
    totol_loss = 0
    y_predicts, y_labels = [], []
    for train_index, test_index in kfold.split(data_x):
        model.train()
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        x_train = x_train.type(torch.FloatTensor).to("cuda")
        y_train = y_train.type(torch.LongTensor).to("cuda")
        x_test = x_test.type(torch.FloatTensor).to("cuda")
        #y_test = y_test.type(torch.LongTensor).to("cuda")
        pred = model(x_train)
        loss = criterion(pred, y_train)
        totol_loss = totol_loss+loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            out_max = outputs.detach()
            out_max = torch.argmax(out_max, dim=1)
            out_max = out_max.to("cpu")
            y_predicts.extend(out_max)
            y_labels.extend(y_test.numpy())
    y_predicts = np.array(y_predicts)
    train_acc,m = model.result(y_predicts, y_labels)
    return train_acc,m

def eval():
    model.eval()
    y_predicts = []
    y_labels = []
    with torch.no_grad():
        outputs = model(test_x)
        out_max = outputs.detach()
        out_max = torch.argmax(out_max, dim=1)
        out_max = out_max.to("cpu")
        y_predicts.extend(out_max)
        y_labels.extend(test_y.numpy())
    y_predicts = np.array(y_predicts)
    dev_acc, m = model.result(y_predicts, y_labels)
    return dev_acc, m

if __name__=="__main__":
    '''
EIIP：41*1
ANF： 41*1
binary：41×4
NCP：41×3
CKSNAP：4×4×6
ENAC2：40×4
PSTNPss:1*39
ram：1*24
    '''
    feature_dim = [[41, 4]]
    feature_name = ["binary+NPC"]
    dims_1 = feature_dim[0]
    dim2_1 = dims_1[0]
    input_size = dims_1[1]
    kfold = KFold(n_splits=10, shuffle=True)
    loo = LeaveOneOut()
    X, Y = read_file()
    data_x, test_x, data_y, test_y = train_test_split(X, Y, test_size=0.1)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    test_x = test_x.type(torch.FloatTensor).to("cuda")
    model = BiLSTM(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00005,
                                 amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    model = model.to("cuda")
    train_acc = 0
    train_m = np.zeros((2, 2))
    dev_acc = 0
    dev_m = np.zeros((2, 2))
    for epoch in range(1000):
        train_acc, train_m = train()
        acc, m = eval()
        if acc > dev_acc:
            dev_acc = acc
            dev_m = m
    print("train_acc:", train_acc)
    print("train_m:", train_m)
    print("dev_acc:", dev_acc)
    print("dev_m:", dev_m)


