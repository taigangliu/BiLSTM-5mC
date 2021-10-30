from model_bilstm import BiLSTM
from model_util import load_model
import os
import numpy as np
import torch
import random
import torch.utils.data as data
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
from get_data import get_pos_feature,get_neg_feature
import csv
amino = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

class MyDataset(data.Dataset):
    def __init__(self, feature1,feature2,labels):
        self.feature1 = feature1
        self.feature2 = feature2
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        feature1,feature2, target = self.feature1[index],self.feature2[index],self.labels[index]
        return feature1,feature2,target

    def __len__(self):
        return len(self.feature1)

def get_loader(mode,feature1,feature2,labels):
    #features,labels = one_hot(sentences,targets)
    if(mode=='train'):
        loader = torch.utils.data.DataLoader(MyDataset(feature1,feature2,labels), batch_size=2048, shuffle=True)
    if (mode == 'test'):
        loader = torch.utils.data.DataLoader(MyDataset(feature1,feature2,labels), batch_size=2048, shuffle=False)
    return loader



if __name__=="__main__":
    seed = 1
    torch.manual_seed(seed)  # 为CPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    csvfile1 = open("label.csv", "w")
    writer1 = csv.writer(csvfile1)

    csvfile2 = open("probs.csv", "w")
    writer2 = csv.writer(csvfile2)
    #-----------------------------------get data------------------------
    pos_data, pos_tag = get_pos_feature("train","one_hot",41,4)
    neg_data, neg_tag = get_neg_feature("train","one_hot",41,4)
    X = np.concatenate((pos_data, neg_data), 0)
    Y = np.concatenate((pos_tag, neg_tag), 0)

    pos_data, pos_tag = get_pos_feature("train", "ncf", 41, 4)
    neg_data, neg_tag = get_neg_feature("train", "ncf", 41, 4)
    X1 = np.concatenate((pos_data, neg_data), 0)


    kflod = KFold(n_splits=5, shuffle=True, random_state=0)
    j = 0
    result_acc = 0
    result_m = np.zeros((2,2))
    result_auc = 0
    y_true = []
    y_predicts = []
    y_probs = []

    for train_index, test_index in kflod.split(X):
        print("------------------------------------------------------")
        tags = []
        y_prob = []
        ncf_train_x = X1[train_index]
        train_y = Y[train_index]
        ncf_test_x = X1[test_index]
        one_hot_test_x = X[test_index]
        test_y = Y[test_index]
        test_loader = get_loader("test", one_hot_test_x,ncf_test_x,test_y)
        # -------------------------------model---------------------------
        train_y_true = []
        test_y_true = []
        for i in range(11):
            print(j,i)
            #model = BiLSTM(4).to("cuda")
            model = BiLSTM(4)
            mode_state_dict = load_model("./model/train_" +str(j)+ os.sep, str(i))
            model.load_state_dict(mode_state_dict)
            model.eval()

            #model1 = BiLSTM(4).to("cuda")
            model1 = BiLSTM(4)
            mode_state_dict1 = load_model("./model/ncf_train_" + str(j)+os.sep,str(i))
            model1.load_state_dict(mode_state_dict1)
            model1.eval()

            tag = []
            tag1 = []
            tests_y = []
            yprob = []
            yprob1 = []
            for test_x1,test_x2,test_y in test_loader:
                with torch.no_grad():
                    #test_x1 = test_x1.type(torch.FloatTensor).to("cuda")
                    test_x1 = test_x1.type(torch.FloatTensor)
                    outputs = model(test_x1)
                    out = outputs.detach()
                    out = torch.softmax(out, dim=-1)
                    yprob.extend(out.to("cpu").numpy())
                    out = torch.argmax(out, dim=1)
                    out = out.to("cpu")
                    tag.extend(out)
                    tests_y.extend(test_y.numpy())

                    #test_x2 = test_x2.type(torch.FloatTensor).to("cuda")
                    test_x2 = test_x2.type(torch.FloatTensor)
                    outputs = model1(test_x2)
                    out = outputs.detach()
                    out = torch.softmax(out, dim=-1)
                    yprob1.extend(out.to("cpu").numpy())
                    out = torch.argmax(out, dim=1)
                    out = out.to("cpu")
                    tag1.extend(out)



            test_y_true = tests_y
            #print(test_y_true)
            tags.append(tag)
            y_prob.append(yprob)
            tags.append(tag1)
            y_prob.append(yprob1)

        # -----------------------------voting----------------------------
        tags = np.array(tags)
        y_prob =  np.array(y_prob)
        result = tags[0]
        probs = y_prob[0]
        for i in range(1, 22):
            result = result + tags[i]
            probs = probs + y_prob[i]
        predicts = []
        for i in range(len(result)):
            if (result[i] < 22):
                predicts.append(0)
            else:
                predicts.append(1)
        probs = probs/22
        y_predicts.extend(predicts)
        y_true.extend(test_y_true)
        y_probs.extend(probs)
        j = j+1

    y_probs = np.array(y_probs)
    writer1.writerow(y_true)
    writer2.writerow(y_probs)

    result_acc = accuracy_score(y_true, y_predicts)
    result_m = confusion_matrix(y_true, y_predicts)
    result_auc = roc_auc_score(y_true,y_probs[:,1])
    print(result_acc)
    print(result_m)
    print(result_auc)
