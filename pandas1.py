import pandas as pd

# data = pd.DataFrame({
#     "Country": ['Russia', 'Colombia', 'Chile', 'Nigeria'],
#     "Rank": [121, 40, 100, 130]
# })

# #We can do a quick analysis of any data set using:
# print(data.describe())
# print()
# print(data.info())


# #Let's create another data frame.
# data = pd.DataFrame({'group':['a', 'a', 'a', 'b','b', 'b', 'c', 'c','c'],'ounces':[4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
# print(data)
# print(data.sort_values(by=['ounces'], ascending=True, inplace=False))
# print(data.sort_values(by=['group', 'ounces'], ascending=True, inplace=False))




# #create another data with duplicated rows
# data = pd.DataFrame({'k1':['one']*3 + ['two']*4, 'k2':[3,2,1,3,3,4,4]})
# print(data)
# print(data.sort_values(by='k2'))
# print(data.drop_duplicates())

# # remove duplicate values from the k1 column
# print(data.drop_duplicates(subset='k1'))


data = pd.DataFrame({
    'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham','nova lox'],
    'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]
    })