from model_bilstm import BiLSTM
#from model_cnn import dynamic_model
from model_util import load_model
import os
import numpy as np
import torch
import random
import torch.utils.data as data
import csv
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from  get_data import get_pos_feature,get_neg_feature
amino = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
torch.set_default_tensor_type(torch.DoubleTensor)


class MyDataset(data.Dataset):
    def __init__(self, feature1,feature2,labels):
        self.feature1 = feature1
        self.feature2 = feature2
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        feature1, feature2,target = self.feature1[index],self.feature2[index],self.labels[index]
        return feature1,feature2, target

    def __len__(self):
        return len(self.feature1)

def get_loader(mode,feature1,feature2,labels):
    #features,labels = one_hot(sentences,targets)
    if(mode=='train'):
        loader = torch.utils.data.DataLoader(MyDataset(feature1,feature2, labels), batch_size=2048, shuffle=True)
    if (mode == 'test'):
        loader = torch.utils.data.DataLoader(MyDataset(feature1,feature2,labels), batch_size=2048, shuffle=False)
    return loader

if __name__=="__main__":
    seed = 1
    torch.manual_seed(seed)  # 为CPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    csvfile1 = open("test_model_one_hot_ncf_label.csv", "w")
    writer1 = csv.writer(csvfile1)

    csvfile2 = open("test_model_one_hot_ncf_probs.csv", "w")
    writer2 = csv.writer(csvfile2)

    tags = []
    y_probs = []
    #-----------------------------------get data------------------------
    test_pos_data1, test_pos_tag = get_pos_feature("test","one_hot",41,4)
    test_neg_data1, test_neg_tag = get_neg_feature("test","one_hot",41,4)
    test_data1 = np.concatenate((test_pos_data1,test_neg_data1),0)

    test_pos_data2, test_pos_tag = get_pos_feature("test","ncf",41,4)
    test_neg_data2, test_neg_tag = get_neg_feature("test","ncf",41,4)
    test_data2 = np.concatenate((test_pos_data2,test_neg_data2),0)


    #test_data = np.concatenate((test_data1,test_data2),-1)

    test_tag = np.concatenate((test_pos_tag,test_neg_tag),0)

    #test_data = test_data2

    test_loader = get_loader("test",test_data1,test_data2,test_tag)

    #-------------------------------model---------------------------
    train_y_true = []
    test_y_true = []
    for i in range(11):

        #model = BiLSTM(4).to("cuda")
        model = BiLSTM(4)
        #model = dynamic_model(41,4,4).to("cuda")
        mode_state_dict = load_model("./model/best_model" + os.sep, str(i))
        model.load_state_dict(mode_state_dict)
        model.eval()

        #model1 = BiLSTM(4).to("cuda")
        model1 = BiLSTM(4)
        mode_state_dict1 = load_model("./model/best_model_ncf" + os.sep, str(i))
        model1.load_state_dict(mode_state_dict1)
        model1.eval()


        tag = []
        tag1 = []
        tests_y = []
        yprob = []
        yprob1 = []
        for test_x1, test_x2,test_y in test_loader:
            with torch.no_grad():
                #test_x1 = test_x1.type(torch.FloatTensor).to("cuda")
                test_x1 = test_x1.type(torch.DoubleTensor)
                outputs = model(test_x1)
                out = outputs.detach()
                out = torch.softmax(out, dim=-1)
                yprob.extend(out.to("cpu").numpy())
                out = torch.argmax(out, dim=1)
                out = out.to("cpu")
                tag.extend(out)
                tests_y.extend(test_y)

                #test_x2 = test_x2.type(torch.FloatTensor).to("cuda")
                test_x2 = test_x2.type(torch.DoubleTensor)
                outputs = model1(test_x2)
                out = outputs.detach()
                out = torch.softmax(out, dim=-1)
                yprob1.extend(out.to("cpu").numpy())
                out = torch.argmax(out, dim=1)
                out = out.to("cpu")
                tag1.extend(out)



        test_y_true = tests_y
        tags.append(tag)
        y_probs.append(yprob)
        tags.append(tag1)
        y_probs.append(yprob1)


    #-----------------------------voting----------------------------
    tags = np.array(tags)
    y_probs = np.array(y_probs)
    result = tags[0]
    probs = y_probs[0]
    for i in range(1,22):
        result = result+tags[i]
        probs = probs+y_probs[i]

    predicts = []
    for i in range(len(result)):
        if(result[i]<22):
            predicts.append(0)
        else:
            predicts.append(1)
    probs = probs/22
    print(len(predicts))
    score = accuracy_score(test_y_true,predicts)
    m = confusion_matrix(test_y_true,predicts)
    auc = roc_auc_score(test_y_true,probs[:,1])
    print(score)
    print(m)
    print(auc)















