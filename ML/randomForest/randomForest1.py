from sklearn import preprocessing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score


# load the data
train = pd.read_csv("./ML/train.csv")
test = pd.read_csv("./ML/test.csv")
for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))

y = train['target']
del train['target']

X = train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# train the RF classifier
clf = RandomForestClassifier(n_estimators = 500, max_depth=6)
clf.fit(X_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=6, max_features='auto', max_leaf_nodes = None, 
                    min_impurity_split = 1e-07, min_samples_leaf = 1, min_samples_split = 2, min_weight_fraction_leaf = 0.0, 
                       n_estimators=500, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

#make prediction and check model's accuracy
prediction = clf.predict(X_test)
acc = accuracy_score(np.array(y_test), prediction)
print('The accuracy of random forest is {}'.format(acc))
