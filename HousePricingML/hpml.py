import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)

import seaborn as sns  # statistical data visualization
from scipy import stats
from scipy.stats import norm

# loading data
train = pd.read_csv("D:\MyProjects\MachineLearningInit\\all\\train.csv")
test = pd.read_csv("D:\MyProjects\MachineLearningInit\\all\\test.csv")
# print(train.head())

# print('The train data has {0} rows and {1} columns'.format(train.shape[0], train.shape[1]))
# print('-----------------------')
# print('The test data has {0} rows and {1} columns'.format(test.shape[0], test.shape[1]))
# train.info()

# print(train.columns[train.isnull().any()])

# missing value counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss>0]
miss.sort_values(inplace=True)
# print(miss)

# visualising missing values
miss = miss.to_frame() # convert to data frame
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

# plot the missing value count
sns.set(style = "whitegrid", color_codes = True)
sns.barplot(x = "Name", y = "count", data = miss)

plt.xticks(rotation = 90) # rotating the labels on x-axis
# plt.show()

# check distibution of target variable
# SalePrice
# print(train["SalePrice"].tolist())
# sns.distplot(train["SalePrice"])
# plt.legend()
# plt.show()

# # skewness
# print("The skewness of saleprice is {}".format(train["SalePrice"].skew()))

# # tranform variable distribution to get closer to normal
# target = np.log(train["SalePrice"])
# print("Skewness is", target.skew())
# sns.distplot(target)
# plt.legend()
# plt.show()

############################################################
# separate numeric and categorical variables and explore this data from a different angle.

# separate variables into new data frames
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1], cat_data.shape[1]))

# remove the Id variable from numeric data.
del numeric_data["Id"]

# # correlation plot
corr = numeric_data.corr()
# sns.heatmap(corr)
# plt.show() # Notice the last row of this map. We can see the correlation of all variables against SalePrice.

# OverallQual feature is 79% correlated with the target variable
# People usually consider these parameters for their dream house

# print(corr["SalePrice"].sort_values(ascending = False)[:15], '\n') # top 15 values
# print("------------------------------------")
# print(corr['SalePrice'].sort_values(ascending = False)[-5:]) #last 5 values

################
# check the OverallQual variable in detail.
print(train["OverallQual"].unique()) #quality is measured on a scale of 1 to 10. Hence, we can fairly treat it as an ordinal variable.


