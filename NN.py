import numpy as np
import pandas as pd
import torch,sklearn
from sklearn import model_selection
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval import metrics

    
    

def load_data(datastr : str,features,target):
    def month_fun(m):
        d = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0}
        d[m] += 1.0
        return d
    def stat_fun(s):
        d = {'USW00012839' : 0, 'USW00014819' :0, 'USW00013904' :0, 'USW00094728': 0}
        d[s] += 1.0
        return d

    dataset = pd.read_csv(datastr)
    dataset = dataset[features+[target,"STATION","Month"]]
    dataset.dropna(inplace=True)
    dataset[target] = dataset[target].astype("float")
    for f in features:
        dataset[f] = dataset[f].astype("float")
    dataset[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]] = \
        dataset.apply(lambda x : month_fun(x["Month"]),axis = 1,result_type="expand")
    dataset[['USW00012839', 'USW00014819', 'USW00013904', 'USW00094728']] =\
            dataset.apply(lambda x : stat_fun(x["STATION"]),axis = 1, result_type="expand")
    dataset.dropna(inplace=True)
    dataset.drop(columns = ["Month","STATION"],inplace = True)
    train,test = model_selection.train_test_split(dataset,test_size = 0.1,random_state = 3)
    print(train.columns)
    train_x = train.drop(columns = [target])
    train_y = train[target]
    test_x = test.drop(columns = [target])
    test_y = test[target]
    return train_x,train_y,test_x,test_y

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.layers = self.get_fc_layers()

    def get_fc_layers(self):

        layers = nn.Sequential(
            nn.Linear(26, 20,dtype = torch.float64),
            nn.ReLU(),
            nn.Linear(20, 14,dtype = torch.float64),
            nn.ReLU(),
            nn.Linear(14, 8,dtype = torch.float64),
            nn.ReLU(),
            nn.Linear(8, 1,dtype = torch.float64),
           
            )
        return layers


    # define forward function
    def forward(self, input):

        x = self.layers(input)
        return x

train_x,train_y,test_x,test_y = load_data("1948_2024.csv",["Week_av","2Week_av","Month_av","d1","d2","d3","d4","d5","d6","d7"],"TMAX")
model = NN()
print(model)
criterion = nn.MSELoss()

# Defining optimizer
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
print(train_x.dtypes)
train_loader = DataLoader(list(zip(torch.tensor(train_x.values),torch.tensor(train_y.values))),batch_size  = 64)
test_loader = DataLoader(list(zip(torch.tensor(test_x.values),torch.tensor(test_y.values))),batch_size  = 64)

import time
metric = metrics.R2Score()
epochs = 10
for epoch in range(epochs):

    # Initialising statistics that we will be tracking across epochs
    start_time = time.time()
    total_correct = 0
    total = 0
    total_loss = 0

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # loading onto cuda if available*


        # zero the parameter gradients: Clean the gradient caclulated in the previous iteration
        optimizer.zero_grad() #Set all graidents to zero for each step as they accumulate over backprop

        # forward + backward + optimize
        inputs = inputs.view(inputs.shape[0], -1)
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        # Calculate gradient of matrix with requires_grad = True
        loss.backward() #computes dloss/dx for every parameter x which has requires_grad=True
        optimizer.step() # x += -lr * x.grad ie updates the weights of the parameters

        # Adding loss to total loss
        total_loss += loss.item()

        # Checking which output label has max probability

        # Tracking number of correct predictions

        # Calculating accuracy, epoch-time
    end_time = time.time() - start_time

    # Printing out statistics
    print("Epoch no.",epoch+1 ,"|accuracy: ", round(0, 3),"%", "|total_loss: ", total_loss, "| epoch_duration: ", round(end_time,2),"sec")

for i, data in enumerate(test_loader, 0):

# everything here is similar to the train function, except we're not calculating loss and not preforming backprop
    inputs, labels = data


    inputs = inputs.view(inputs.shape[0], -1)
    outputs = model.forward(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.shape[0]
    correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct/total}%')

