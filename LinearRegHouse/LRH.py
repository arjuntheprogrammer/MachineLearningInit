import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

# import Boston dataset from datasets library
from sklearn.datasets import load_boston
boston = load_boston()

# print(boston)

# transferring to dataFrame
df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)
# print(df_x)
# print(df_y)


# training the regression model
reg = linear_model.LinearRegression()

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)

# fitting the data into the model
reg.fit(x_train, y_train)

# Calculating coefficients
# print(reg.coef_)

a = reg.predict(x_test)
# print(a)
# print(y_test)

# finding the mean square error(MSE)
print(np.mean((a-y_test)**2))

