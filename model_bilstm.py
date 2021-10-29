import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report,precision_score,recall_score,confusion_matrix
import numpy as np
from torch.nn import CrossEntropyLoss
from sklearn.svm import SVC
class BiLSTM(nn.Module):
    def __init__(self,input_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=300, bidirectional=True)
        # fc
        self.fc = nn.Sequential(nn.Linear(300*2, 2))

    def forward(self,X):
        batch_size = X.shape[0]
        # X: [batch_size, max_len, n_class]
        input = X.transpose(0, 1)
        #print(X.size())
        hidden_state = torch.randn(1*2, batch_size, 300).to("cuda")   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1*2, batch_size, 300).to("cuda")     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        #print(outputs.to("cpu").detach().numpy())
        #model = outputs.to("cpu").detach().numpy()
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model


    def result(self, y_pred, y_true):
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/len(y_pred)
        con_matix=confusion_matrix(y_true,y_pred)
        return acc,con_matix

#model = BiLSTM()
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)
