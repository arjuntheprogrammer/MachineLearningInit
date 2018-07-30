import pandas as pd
import numpy as np

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

# ##############################################################################################################
# data = pd.DataFrame({
#     'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham','nova lox'],
#     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]
#     })

# meat_to_animal = {
#     'bacon': 'pig',
#     'pulled pork': 'pig',
#     'pastrami': 'cow',
#     'corned beef': 'cow',
#     'honey ham': 'pig',
#     'nova lox': 'salmon'
# }


# def meat_2_animal(series):
#     if series['food'] == 'bacon':
#         return 'pig'
#     elif series['food'] == 'pulled pork':
#         return 'pig'
#     elif series['food'] == 'pastrami':
#         return 'cow'
#     elif series['food'] == 'corned beef':
#         return 'cow'
#     elif series['food'] == 'honey ham':
#         return 'pig'
#     else:
#         return 'salmon'


# data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
# # print(data)

# # another way of doing it
# lower = lambda x: x.lower()
# data['food'] = data['food'].apply(lower)
# data['animal2'] = data.apply(meat_2_animal, axis = 'columns')
# # print(data)

# data = data.assign(new_variable = data['ounces']*10)
# # print(data)

# data.drop('animal2', axis='columns', inplace = True)
# print(data)

# data = pd.Series([1., -999., 2., -999., -1000., 3.])
# print(data)

# #We can also replace multiple values at once.
# data.replace(-999, np.nan, inplace=True)
# print(data)


data = pd.DataFrame(np.arange(12).reshape((3,4)), index = ['Ohio', 'Colorado', 'New York'], columns = ['one', 'two', 'three', 'four'])
print(data)
