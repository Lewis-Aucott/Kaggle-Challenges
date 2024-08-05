#Import required packages
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import pandas as pd


#Read in data source
training_data = np.loadtxt('data/train.csv',delimiter=",",dtype=np.float32,skiprows=1)
train_x=torch.from_numpy(xy[:, 1:])
train_y=torch.from_numpy(xy[:, [0]])
n_samples=xy.shape[0]

print(train_x)
# train_set=DataLoader(training_data, batch_size=64, shuffle=True)
    
# train_features=next(iter(train_set))

# print(train_features)