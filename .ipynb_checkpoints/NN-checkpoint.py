import numpy as np
import pandas as pd
import torch,sklearn
from sklearn import model_selection


class NN:
    def __init__(self,features,target):
        self.features = features
        self.target = target

    def month_fun(self,m):
            d = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0}
            d[m] += 1
            return d
    def stat_fun(self,s):
        d = {'USW00012839' : 0, 'USW00014819' :0, 'USW00013904' :0, 'USW00094728': 0}
        d[s] += 1
        return d
    

    def load_data(self,datastr : str):

        self.dataset = pd.read_csv(datastr)
        self.dataset = self.dataset[self.features+[self.target,"STATION","Month"]]
        self.dataset.dropna(inplace=True)
        self.feature_data = pd.DataFrame()
        self.target_data = pd.DataFrame()
        self.dataset[self.target] = self.dataset[self.target].astype("int")
        for f in self.features:
            self.dataset[f] = self.dataset[f].astype("float")
        self.dataset[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]] = \
            self.dataset.apply(lambda x : self.month_fun(x["Month"]),axis = 1,result_type="expand")
        self.dataset[['USW00012839', 'USW00014819', 'USW00013904', 'USW00094728']] =\
              self.dataset.apply(lambda x : self.stat_fun(x["STATION"]),axis = 1, result_type="expand")
        self.dataset.dropna(inplace=True)
        train,test = model_selection.train_test_split(self.dataset,test_size = 0.1,random_state = 3)
        self.train_x = train.drop([self.target])
        self.train_y = train[self.target]
        self.test_x = test.drop([self.target])
        self.test_y = test[self.target]

