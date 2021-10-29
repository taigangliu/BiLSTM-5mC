
# coding: utf-8

# In[ ]:


import sklearn
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import pandas as pd
import sklearn.metrics
import itertools
import pickle
import argparse
import os,sys,re
from collections import Counter
from keras.models import load_model
import xgboost as xgb


# In[ ]:


data_test = pd.read_csv("./test_data.csv",header=None)
data_test = data_test.loc[:,1]
print(data_test[0:4])

A = 'A,C,G,T'
A = A.split(',')
dict_A = dict(zip(A,range(4)))

def NCP(seq):
    AA = 'ACGT'
    probMatr = np.zeros((len(seq),3))
    for j in range(len(seq)):
        if(seq[j] == "A"):
            probMatr[j][0] = 1
            probMatr[j][1] = 1
            probMatr[j][2] = 1
        elif(seq[j] == "C"):
            probMatr[j][0] = 0
            probMatr[j][1] = 1
            probMatr[j][2] = 0
        elif(seq[j] == "G"):
            probMatr[j][0] = 1
            probMatr[j][1] = 0
            probMatr[j][2] = 0
        elif(seq[j] == "T"):
            probMatr[j][0] = 0
            probMatr[j][1] = 0
            probMatr[j][2] = 1    
    return probMatr.reshape(1,-1)

test_ncp = NCP(data_test.loc[1])
for j in range(1,data_test.shape[0]):
    test_ncp = np.vstack((test_ncp,NCP(data_test.loc[j])))
print(test_ncp.shape)


#feature to csv
test_ncp = pd.DataFrame(test_ncp)
test_ncp.to_csv("./5mc_test_ncp.csv",header = None,index = 0)


# In[ ]:


def ANF(sequences):
    nd_feature = []
    for aaindex, aa in enumerate(sequences):
        nd_feature.append(sequences[0: aaindex + 1].count(sequences[aaindex]) / (aaindex + 1))
    return nd_feature


test_anf = ANF(data_test.loc[1])
for j in range(1,data_test.shape[0]):
    test_anf = np.vstack((test_anf,ANF(data_test.loc[j])))
print(test_anf.shape)


#feature to csv
test_anf = pd.DataFrame(test_anf)
test_anf.to_csv("./5mc_test_anf.csv",header = None,index = 0)

