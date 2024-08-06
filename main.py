#Import required packages
import numpy as np
import pandas as pd
import math
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader



#Read in data source
df = pd.read_csv('data/train.csv')

# Drop some columns and separarate into x/y values
train_y=df.pop('Exited')
train_X=df.drop(['id','CustomerId','Surname','Geography'],axis=1)
print(train_X.columns)
print(df)

# Encoding and scaling

def genderEncoding(gender):
  if (gender=='male'):
    return 1
  else:
    return 0
    
train_X.Gender=train_X.Gender.apply(genderEncoding)
train_X.Gender.apply(genderEncoding)
print(train_X.Gender)






# Wrap iterable to iterate over epoch

# train_set=DataLoader(training_data, batch_size=64, shuffle=True)

# train_features=next(iter(train_set))

# print(train_features)