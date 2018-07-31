from  sklearn import preprocessing
import pandas as pd
import numpy as np

# load the data
train = pd.read_csv("./ML/train.csv")
test = pd.read_csv("./ML/test.csv")

# convert the character variable into numeric
for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))
print(train.head())

#<50K = 0 and >50K = 1
print(train.target.value_counts())
