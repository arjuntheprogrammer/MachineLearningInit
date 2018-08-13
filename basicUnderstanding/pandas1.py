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


# data = pd.DataFrame(np.arange(12).reshape((3,4)), index = ['Ohio', 'Colorado', 'New York'], columns = ['one', 'two', 'three', 'four'])
# print(data)

# # Using rename function
# data.rename(index = {'Ohio': 'SanF'}, columns = {'one': 'one_p', 'two': 'two_p'}, inplace=True)
# print(data)

# # Can also use string functions
# data.rename(index = str.upper, columns =str.title, inplace = True )
# print(data)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Categorize
# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# # We'll divide the ages into bins such as 18-25, 26-35,36-60 and 60 and above.
# bins = [18, 25, 35, 60, 100]
# cats = pd.cut(ages, bins)
# print(cats)

# #To include the right bin value, we can do:
# print(pd.cut(ages, bins, right=False))

# #pandas library intrinsically assigns an encoding to categorical variables.
# # print(cats.lables)

# #Let's check how many observations fall under each bin
# print(pd.value_counts(cats))

# bin_names = ['Youth', 'YoungAdult', 'MiddleAge', 'Senior']
# new_cats = pd.cut(ages, bins, labels = bin_names)
# print(pd.value_counts(new_cats))

# # We can also calculate cumulative sums
# print()
# print(pd.value_counts(new_cats).cumsum())


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
# #  Grouping data and creating pivots in pandas

# # df = pd.DataFrame({
# #     'key1': ['a', 'a', 'b', 'b', 'a'],
# #     'key2': ['one', 'two', 'one', 'two', 'one'],
# #     'data1': np.random.randn(5),
# #     'data2': np.random.randn(5) })
# # print(df)

# # print("")
# # # calculate the mean of data1 column by key1
# # grouped = df['data1'].groupby(df['key1'])
# # print(grouped.mean())

# #  Slice the data frame
# dates = pd.date_range('20130101', periods=6)
# df = pd.DataFrame(np.random.randn(6,4), index = dates, columns = list('ABCD'))

# print(df)

# #get first n rows from the data frame
# print(df[:3])

# #slice based on date range
# print(df['20130101':'20130104'])

# #slicing based on column names
# print(df.loc[:, ['A', 'B']])

# #slicing based on both row index labels and column names
# print(df.loc['20130102':'20130103', ['A', 'B']])

# #slicing based on index of columns
# print(df.iloc[3]) #return 4th row

# #returns a specific range of rows
# print(df.iloc[2:4, 0:2])

# # Boolean indexing based on column values
# print(df[df.A > 1])

# # we can copy the data set
# df2 = df.copy()
# df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
# print(df2)

# # select rows based on column vaulues
# print(df2[df2['E'].isin(['two', 'four'])])

# # select all rows except those woth two and four
# print(df2[~df2['E'].isin(['two', 'four'])])

# # list all columns where A is greater than C
# print(df.query('A > C'))

# # using or condition
# print(df.query('A < B | C > A'))

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# #  Pivot tables
# create data frame
data = pd.DataFrame({
    'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
    'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]
})

print(data)

# calculate means of each group
print(data.pivot_table(values='ounces', index = 'group', aggfunc=np.mean))


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##



