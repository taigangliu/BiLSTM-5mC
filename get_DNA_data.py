import csv
import numpy as np
import os
amino={'A':0,'C':1,'G':2,'T':3}
def get_labels(file_path):
    labels = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels.append(float(row[0]))
    return labels


def get_data_list():
    sentences = []
    tags = get_labels('5mC_features' + os.sep + 'DNA_label_train.csv')
    with open('5mC_features' + os.sep + "DNA_train.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            sentence = [x for x in row[0]]
            sentences.append(sentence)
    return sentences,tags


def read_file():
    sentences,target = get_data_list()
    feature1 = []
    feature2 = []
    feature3 = []
    for sentence in sentences:
        input = [amino[n] for n in sentence]
        feature1.append(np.eye(4)[input])

    with open('5mC_features'+os.sep+'NPF.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input = [float(x) for x in row]
            feature2.append(input)

    with open('5mC_features'+os.sep+'One_hot.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input = [float(x) for x in row]
            feature3.append(input)
    
    #feature1 = np.array(feature1)
    #feature2 = np.array(feature2)
    #feature3 = np.array(feature3)
    #feature2 = feature2.reshape(feature2.shape[0],41,-1)
    #feature3 = feature3.reshape(feature3.shape[0],41,-1)
    #将列表转换为tensor数据
    #feature = np.concatenate((feature3,feature2),axis=2)
    #feature = np.array(feature)

    feature = np.array(feature1)
    target = np.array(target)
    return feature, target

def get_data(file_name,dim1,dim2):
    _,target = get_data_list()
    feature = []
    with open('5mC_features' + os.sep + file_name+'.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            input = [float(x) for x in row]
            feature.append(input)
    feature = np.array(feature)
    feature = feature.reshape(feature.shape[0], dim1, dim2)
    target = np.array(target)
    return feature,target
#feature,tag=read_file()
#feature,tag = get_data('AAC',1,4)
#print(feature.shape)
#print(tag.shape)