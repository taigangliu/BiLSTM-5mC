
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import math
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import xgboost
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import BaseDiscreteNB, MultinomialNB,BernoulliNB
from sklearn.model_selection import LeaveOneOut
from Bio.Seq import Seq


data_train = pd.read_csv("./dataset/train.csv",header=None)
data_train = data_train.loc[:,0]


A = 'A,C,G,T'
A = A.split(',')
dict_A = dict(zip(A,range(4)))
print(dict_A)


def One_hot(seq):
    hot_code = np.zeros((len(seq),4))
    for i in range(len(seq)):
        hot_code[i][dict_A[seq[i]]] = 1
    return hot_code.reshape(1,-1)


train_hot = One_hot(data_train.loc[0])
for j in range(1,data_train.shape[0]):
    train_hot = np.vstack((train_hot,(One_hot(data_train.loc[j]))))
print(train_hot.shape)


#feature to csv
train_hot = pd.DataFrame(train_hot)
train_hot.to_csv("./dataset/train_hot.csv",header = None,index = 0)


X_train = train_hot
y_train = pd.read_csv("./dataset/label_train.csv",header = None)
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)

