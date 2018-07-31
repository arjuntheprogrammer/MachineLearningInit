import pandas as pd
import numpy as np

# load the data
train = pd.read_csv("./ML/train.csv")
test = pd.read_csv("./ML/test.csv")
# train.info()
# print('The train data has', train.shape)
# print('The test data has', test.shape)

# # lets have a glimpse of data set
# print(train.head())

# nans = train.shape[0] - train.dropna().shape[0]
# print("%d rows have missing values in the train data" %nans)

# nand = test.shape[0] - test.dropna().shape[0]
# print("%d rows have missing values in the test data" %nand)

# #  which columns have missing values.
# print(train.isnull().sum())

# # count the number of unique values from character variables.
# cat = train.select_dtypes(include=['O'])
# print(cat.apply(pd.Series.nunique))

# impute missing values with their respective mode - workclass, occupationm native 
# Workclass
# print(train.workclass.value_counts(sort = True))
train.workclass.fillna("Private", inplace = True)

# occupation
# print(train.occupation.value_counts(sort = True))
train.occupation.fillna("Prof-speciality", inplace = True)

# Workclass
# print(train['native.country'].value_counts(sort = True))
train['native.country'].fillna("United-States", inplace=True)

# print(train.isnull().sum())

###########################################
#check proportion of target variable
# print(train.target.value_counts()/train.shape[0])

print(pd.crosstab(train.education, train.target, margins= True)/train.shape[0])

