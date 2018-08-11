# https://www.hackerearth.com/practice/machine-learning/machine-learning-projects/python-project/tutorial/

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

# # missing value counts in each of these columns
# miss = train.isnull().sum()/len(train)
# miss = miss[miss>0]
# miss.sort_values(inplace=True)
# # print(miss)

# # visualising missing values
# miss = miss.to_frame() # convert to data frame
# miss.columns = ['count']
# miss.index.names = ['Name']
# miss['Name'] = miss.index

# # plot the missing value count
# sns.set(style = "whitegrid", color_codes = True)
# sns.barplot(x = "Name", y = "count", data = miss)

# plt.xticks(rotation = 90) # rotating the labels on x-axis
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
# numeric_data = train.select_dtypes(include=[np.number])
# cat_data = train.select_dtypes(exclude=[np.number])
# print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1], cat_data.shape[1]))

# # remove the Id variable from numeric data.
# del numeric_data["Id"]

# # correlation plot
# corr = numeric_data.corr()
# sns.heatmap(corr)
# plt.show() # Notice the last row of this map. We can see the correlation of all variables against SalePrice.

# OverallQual feature is 79% correlated with the target variable
# People usually consider these parameters for their dream house

# print(corr["SalePrice"].sort_values(ascending = False)[:15], '\n') # top 15 values
# print("------------------------------------")
# print(corr['SalePrice'].sort_values(ascending = False)[-5:]) #last 5 values

################
# # check the OverallQual variable in detail.
# print(train["OverallQual"].unique()) #quality is measured on a scale of 1 to 10. Hence, we can fairly treat it as an ordinal variable.

# # check mean price per quality and plot it
# pivot = train.pivot_table(index  = "OverallQual", values="SalePrice", aggfunc=np.median)
# # print(pivot.sort_index())
# pivot.plot(kind="bar", color="red")
# # plt.legend()
# # plt.show()


# #GrLivArea variable
# sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'])
# plt.legend()
# plt.show()  # we see a direct correlation of living area with sale price

# # simplest way to understand categorical variables
# print(cat_data.describe())

# # lets check median sales price based on saleCondition
# sp_pivot = train.pivot_table(index="SaleCondition", values="SalePrice", aggfunc=np.median)
# print(sp_pivot)

# sp_pivot.plot(kind='bar', color='red')
# plt.legend()
# plt.show()

"""
we used correlation to determine the influence of numeric features on SalePrice. Similarly, 
we'll use the ANOVA test to understand the correlation between categorical variables and SalePrice. 
ANOVA test is a statistical technique used to determine if there exists a significant difference in the mean of groups. 
For example, let's say we have two variables A and B. Each of these variables has 3 levels(a1, a2, a3 and b1, b2, b3). 
If the mean of these levels with respect to the target variable is the same, 
the ANOVA test will capture this behavior and we can safely remove them.
While using ANOVA, our hypothesis is as follows:
Ho - There exists no significant difference between the groups. Ha - There exists a significant difference between the groups.
"""

"""
scipy.stats.mstats.f_oneway(*args)[source]
Performs a 1-way ANOVA, returning an F-value and probability given any number of groups. From Heiman, pp.394-7.

Usage: f_oneway(*args), where *args is 2 or more arrays, one per treatment group.

Returns:	
statistic : float
The computed F-value of the test.

pvalue : float
The associated p-value from the F-distribution.
"""

# cat = [f for f in train.columns if train.dtypes[f] == 'object']

# def anova(frame):
#     anv = pd.DataFrame()
#     anv['features'] = cat
#     pvals = []

#     f=0
#     for c in cat:
#         samples = []
#         for cls in frame[c].unique():
#             s = frame[frame[c] == cls]["SalePrice"].values
#             samples.append(s)
        
#         pval = stats.f_oneway(*samples)[1]
#         pvals.append(pval)

#     anv['pval'] = pvals
#     return anv.sort_values("pval")


# cat_data["SalePrice"] = train.SalePrice.values
# k = anova(cat_data)
# k['disparity'] = np.log(1./k['pval'].values)
# sns.barplot(data = k, x = 'features', y = 'disparity')
# plt.xticks(rotation = 90)
# plt.show() # among all categorical variablesNeighborhoodturned out to be the most important feature

#############################
# create numeric plots
# num = [f for f in train.columns if train.dtypes[f] != 'object']
# num.remove('Id')
# nd = pd.melt(train, value_vars = num)
# n1 = sns.FacetGrid(nd, col='variable', col_wrap = 4, sharex=False, sharey=False)
# n1 = n1.map(sns.distplot, 'value')
# plt.show()  # most of the variables are right skewed

# """
# pd.melt
# This function is useful to massage a DataFrame into a format where one or more columns are identifier variables (id_vars), 
# while all other columns, considered measured variables (value_vars), are unpivoted to the row axis, 
# leaving just two non-identifier columns, variable and value.
# """

# # create boxplots for visualizing categorical variables.

# def boxplot(x,y,**kwargs):
#     sns.boxplot(x=x, y=y)
#     x= plt.xticks(rotation=90)

# cat = [f for f in train.columns if train.dtypes[f] == 'object']

# p = pd.melt(train, id_vars = "SalePrice", value_vars = cat)
# g = sns.FacetGrid(p, col='variable', col_wrap = 2, sharex=False, sharey=False, size = 5)
# g = g.map(boxplot, 'value', 'SalePrice')
# plt.show()

# #######################################################################################
# Data Pre-Processing

# removing outliers
train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)
print(train.shape) # removed 4 rows

"""In row 666, in the test data, it was found that information in variables related to 'Garage' (GarageQual, GarageCond, GarageFinish, GarageYrBlt) 
is missing. Let's impute them using the mode of these respective variables.
"""
# imputing using mode

test.loc[666, 'GarageQual'] = "TA" # stats.mode(test["GarageQual"]).mode
test.loc[666, 'GarageCond'] = "TA" # stats.mode(test["GarageCond"]).mode
test.loc[666, 'GarageFinish'] = "Unf" # stats.mode(test["GarageFinish"]).mode
test.loc[666, 'GarageYrBlt'] = "1980" # stats.mode(test["GarageYrBlt"]).mode

"""
In row 1116, in test data, all garage variables are NA except GarageType. Let's mark it NA as well.
"""
# mark as missing
test.loc[1116, "GarageType"] = np.nan



# importing function
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def factorize(data, var, fill_na = None):
    if fill_na is not None:
        data[var].fillna(fill_na, inplace=True)

    le.fit(data[var])
    data[var] = le.transform(data[var])
    return data

# combine the data set
alldata = train.append(test)
print(alldata.shape)

# impute lotfrontage by median of neighborhood
lot_forntage_by_neighborhood = train["LotFrontage"].groupby(train["Neighborhood"])

for key, group in lot_forntage_by_neighborhood:
    idx = (alldata["Neighborhood"] == key) & (alldata["LotFrontage"].isnull())
    alldata.loc[idx, 'LotFrontage'] = group.median()

#imputing missing values
alldata["MasVnrArea"].fillna(0, inplace=True)
alldata["BsmtFinSF1"].fillna(0, inplace=True)
alldata["BsmtFinSF2"].fillna(0, inplace=True)
alldata["BsmtUnfSF"].fillna(0, inplace=True)
alldata["TotalBsmtSF"].fillna(0, inplace=True)
alldata["GarageArea"].fillna(0, inplace=True)
alldata["BsmtFullBath"].fillna(0, inplace=True)
alldata["BsmtHalfBath"].fillna(0, inplace=True)
alldata["GarageCars"].fillna(0, inplace=True)
alldata["GarageYrBlt"].fillna(0.0, inplace=True)
alldata["PoolArea"].fillna(0, inplace=True)


# we'll convert the categorical variables into ordinal variables
qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])

for i in name:
    alldata[i] = alldata[i].map(qual_dict).astype(int)

alldata["BsmtExposure"] = alldata["BsmtExposure"].map({np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
alldata["BsmtFinType1"] = alldata["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
alldata["BsmtFinType2"] = alldata["BsmtFinType2"].map(bsmt_fin_dict).astype(int)
alldata["Functional"] = alldata["Functional"].map({np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

alldata["GarageFinish"] = alldata["GarageFinish"].map({np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)
alldata["Fence"] = alldata["Fence"].map({np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)

# encoding data
alldata["CentralAir"] = (alldata["CentralAir"] == "Y") * 1.0
varst = np.array(['MSSubClass','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Foundation','SaleCondition'])

for x in varst:
    factorize(alldata, x)

#encode variables and impute missing values
alldata = factorize(alldata, "MSZoning", "RL")
alldata = factorize(alldata, "Exterior1st", "Other")
alldata = factorize(alldata, "Exterior2nd", "Other")
alldata = factorize(alldata, "MasVnrType", "None")
alldata = factorize(alldata, "SaleType", "Oth")


# #######################################################################################
# Feature Engineering(adding new features)

#creating new variable (1 or 0) based on irregular count levels
#The level with highest count is kept as 1 and rest as 0
alldata["IsRegularLotShape"] = (alldata["LotShape"] == "Reg") * 1
alldata["IsLandLevel"] = (alldata["LandContour"] == "Lvl") * 1
alldata["IsLandSlopeGentle"] = (alldata["LandSlope"] == "Gtl") * 1
alldata["IsElectricalSBrkr"] = (alldata["Electrical"] == "SBrkr") * 1
alldata["IsGarageDetached"] = (alldata["GarageType"] == "Detchd") * 1
alldata["IsPavedDrive"] = (alldata["PavedDrive"] == "Y") * 1
alldata["HasShed"] = (alldata["MiscFeature"] == "Shed") * 1
alldata["Remodeled"] = (alldata["YearRemodAdd"] != alldata["YearBuilt"]) * 1

#Did the modeling happen during the sale year?
alldata["RecentRemodel"] = (alldata["YearRemodAdd"] == alldata["YrSold"]) * 1

# Was this house sold in the year it was built?
alldata["VeryNewHouse"] = (alldata["YearBuilt"] == alldata["YrSold"]) * 1
alldata["Has2ndFloor"] = (alldata["2ndFlrSF"] == 0) * 1
alldata["HasMasVnr"] = (alldata["MasVnrArea"] == 0) * 1
alldata["HasWoodDeck"] = (alldata["WoodDeckSF"] == 0) * 1
alldata["HasOpenPorch"] = (alldata["OpenPorchSF"] == 0) * 1
alldata["HasEnclosedPorch"] = (alldata["EnclosedPorch"] == 0) * 1
alldata["Has3SsnPorch"] = (alldata["3SsnPorch"] == 0) * 1
alldata["HasScreenPorch"] = (alldata["ScreenPorch"] == 0) * 1

#setting levels with high count as 1 and the rest as 0
#you can check for them using the value_counts function
alldata["HighSeason"] = alldata["MoSold"].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
alldata["NewerDwelling"] = alldata["MSSubClass"].replace({20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

print(alldata.shape)

#create alldata2
alldata2 = train.append(test)
alldata["SaleCondition_PriceDown"] = alldata2.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

# house complete berfore sale or not
alldata["BoughtOffPlan"] = alldata2.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
alldata["BadHeating"] = alldata2.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

# calculating total area using all area columns
area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']

alldata["TotalArea"] = alldata[area_cols].sum(axis=1)
alldata["TotalAraa1st2nd"] = alldata["1stFlrSF"] + alldata["2ndFlrSF"]
alldata["Age"] = 2010 - alldata["YrSold"]
alldata["SeasonSold"] = alldata["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)
alldata["YearsSinceRemodel"] = alldata["YrSold"] - alldata["YearRemodAdd"]

alldata["SeasonSold"] = alldata["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)
alldata["YearsSinceRemodel"] = alldata["YrSold"] - alldata["YearRemodAdd"]

# Simplifications of existing features into bad/average/good based on counts
alldata["SimplOverallQual"] = alldata.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
alldata["SimplOverallCond"] = alldata.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
alldata["SimplPoolQC"] = alldata.PoolQC.replace({1 : 1, 2 : 1, 3 : 2, 4 : 2})
alldata["SimplGarageCond"] = alldata.GarageCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplGarageQual"] = alldata.GarageQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplFireplaceQu"] = alldata.FireplaceQu.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplFireplaceQu"] = alldata.FireplaceQu.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplFunctional"] = alldata.Functional.replace({1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})
alldata["SimplKitchenQual"] = alldata.KitchenQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplHeatingQC"] = alldata.HeatingQC.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplBsmtFinType1"] = alldata.BsmtFinType1.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
alldata["SimplBsmtFinType2"] = alldata.BsmtFinType2.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
alldata["SimplBsmtCond"] = alldata.BsmtCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplBsmtQual"] = alldata.BsmtQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplExterCond"] = alldata.ExterCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
alldata["SimplExterQual"] = alldata.ExterQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})


# # grouping neighborhood variable based on this plot
# train["SalePrice"].groupby(train["Neighborhood"]).median().sort_values().plot(kind='bar')
# plt.show()
# The graph above gives us a good hint on how to combine levels of the neighborhood variable into fewer levels.

neighborhood_map = {"MeadowV" : 0, "IDOTRR" : 1, "BrDale" : 1, "OldTown" : 1, "Edwards" : 1, "BrkSide" : 1, "Sawyer" : 1, "Blueste" : 1, "SWISU" : 2, "NAmes" : 2, "NPkVill" : 2, "Mitchel" : 2, "SawyerW" : 2, "Gilbert" : 2, "NWAmes" : 2, "Blmngtn" : 2, "CollgCr" : 2, "ClearCr" : 3, "Crawfor" : 3, "Veenker" : 3, "Somerst" : 3, "Timber" : 3, "StoneBr" : 4, "NoRidge" : 4, "NridgHt" : 4}

alldata["NeighborhoodBin"] = alldata2["Neighborhood"].map(neighborhood_map)
alldata.loc[alldata2.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
alldata.loc[alldata2.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
alldata.loc[alldata2.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
alldata.loc[alldata2.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
alldata.loc[alldata2.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1

alldata["Neighborhood_Good"].fillna(0, inplace=True)
alldata["SaleCondition_PriceDown"] = alldata2.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

# House completed before sale or not
alldata["BoughtOffPlan"] = alldata2.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
alldata["BadHeating"] = alldata2.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})
print(alldata.shape)


#####################
# create new data
train_new = alldata[alldata["SalePrice"].notnull()]
test_new = alldata[alldata["SalePrice"].isnull()]

print("Train", train_new.shape)
print('----------------------')
print("Test", test_new.shape)


"""Now, we'll transform numeric features and remove their skewness.
"""
numeric_features = [f for f in train_new.columns if train_new[f].dtype != 'object']

# tranform the numeric features using log(x+1)
from scipy.stats import skew
skewed = train_new[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
train_new[skewed] = np.log1p(train_new[skewed])
test_new[skewed] = np.log1p(test_new[skewed])
del test_new["SakePrice"]

# Now, we'll standardize the numeric features.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_new[numeric_features])
scaled = scaler.transform(train_new[numeric_features])

for i, col in enumerate(numeric_features):
    train_new[col] = scaled[:, i]

numeric_features.remove("SalePrice")

scaled = scaler.fit_transform(test_new[numeric_features])
for i, col in enumerate(numeric_features):
    test_new[col] = scaled[:, i]

#  In one-hot encoding, every level of a categorical variable results in a new variable with binary values (0 or 1).
def onehot(onehot_df, df, column_name, fill_na):
    onehot_df[column_name] = df[column_name]
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)
    
    dummies = pd.get_dummies(onehot_df[column_name], prefix="_"+column_name)
    onehot_df = onehot_df.join(dummies)
    onshot_df = onehot_df.drop([column_name], axis = 1)
    return onehot_df


