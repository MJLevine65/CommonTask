import pandas as pd
import os
import numpy as np
from sklearn import linear_model

class LinearModel():
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
        # self.dataset["SNOW"].fillna(0,inplace = True)
        # self.dataset["SNWD"].fillna(0,inplace = True)
        self.dataset.dropna(inplace=True)
        self.feature_data = pd.DataFrame()
        self.target_data = pd.DataFrame()
        self.target_data[self.target] = self.dataset[self.target].astype("int")
        for f in self.features:
            self.feature_data[f] = self.dataset[f].astype("float")
        self.feature_data[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]] = \
            self.dataset.apply(lambda x : self.month_fun(x["Month"]),axis = 1,result_type="expand")
        self.feature_data[['USW00012839', 'USW00014819', 'USW00013904', 'USW00094728']] =\
              self.dataset.apply(lambda x : self.stat_fun(x["STATION"]),axis = 1, result_type="expand")
        self.feature_data.dropna(inplace=True)

    
    def train(self):
        self.model = linear_model.LinearRegression()
        self.ridge_model = linear_model.Ridge(alpha=9999999)
        print(self.feature_data.columns)
        self.model.fit(self.feature_data,self.target_data)
        self.ridge_model.fit(self.feature_data,self.target_data)
        print("Ordinary")
        print(self.model.score(self.feature_data,self.target_data))
        print(self.model.coef_)
        print(self.model.intercept_)
        print("Ridge")
        print(self.model.score(self.feature_data,self.target_data))
        print(self.model.coef_)
        print(self.model.intercept_)

    
    def predict(self, data):
        for i,row in data.iterrows():
            track = list(row["days"])
            data.at[i,"Week_av"] = np.sum(track[-7:])/7
            data.at[i,"2Week_av"] = np.sum(track[-14:])/14
            data.at[i,"Month_av"] = np.sum(track)/30
            data.at[i,"d1"],data.at[i,"d2"],data.at[i,"d3"],data.at[i,"d4"],data.at[i,"d5"],data.at[i,"d6"],data.at[i,"d7"] = track[-7:]
        data[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]] = \
            data.apply(lambda x : self.month_fun(x["Month"]),axis = 1,result_type="expand")
        data[['USW00012839', 'USW00014819', 'USW00013904', 'USW00094728']] =\
              data.apply(lambda x : self.stat_fun(x["STATION"]),axis = 1, result_type="expand")
        data.drop(["Month","STATION","days"],axis = 1,inplace = True)
        return self.model.predict(data),self.ridge_model.predict(data)

lm = LinearModel(["Week_av","2Week_av","Month_av","d1","d2","d3","d4","d5","d6","d7"],"TMAX")
lm.load_data("1948_2024.csv")
lm.train()
#46,75,47,68
NY_daily = [55,60,56,48,41,38,39,43,36,40,41,40,43,45,47,44,41,55,55,62,43,48,55,68,59,49,53,54,57,49]
M_daily =  [76,78,80,85,82,80,78,79,84,76,70,70,74,77,79,76,77,76,78,80,81,79,80,83,85,83,82,86,85,86]
C_daily =  [58,43,43,46,42,48,46,33,30,47,52,58,68,61,52,36,62,72,74,53,46,47,60,74,72,45,50,50,47,47]
A_daily =  [75,66,60,61,67,68,69,73,52,54,68,79,77,87,75,79,83,82,80,70,49,68,81,80,83,91,83,75,83,62]
print(lm.predict(pd.DataFrame({"STATION":['USW00012839', 'USW00014819', 'USW00013904', 'USW00094728'],"Month":[3,3,3,3],"days" : [M_daily,C_daily,A_daily,NY_daily]})))
