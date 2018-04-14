# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:27:33 2018

@author: Ithier
"""

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV
# https://github.com/frc1418/tbapy/blob/master/tbapy/main.py
base_url= "https://www.thebluealliance.com/api/v3/"
auth_key = 'fge7icVbwIkRUkYKFb7Bj045jGELlWspOnCTxJnhkC9jqiLRjE0VBR4ACcez4vxo'

# https://www.dataquest.io/blog/python-api-tutorial/
# https://www.digitalocean.com/community/tutorials/how-to-use-web-apis-in-python-3
# https://www.dataquest.io/blog/python-api-tutorial/
# https://www.dataquest.io/blog/python-api-tutorial/
headers = {"X-TBA-Auth-Key": auth_key}

path = 'C:/Users/Ithier/Documents/CSCI 29/Grad Project/FIRST/Data/'

name = path + 'TotalData.csv'
data = pd.DataFrame.from_csv(name)
data = data.loc[(data["level"] == "ef") | (data["level"] == "qf") | (data["level"] == "sf") | (data["level"] == "f") ]

X = data.drop(["b_score", "r_score", "level", "win_margin"], axis = 1)
y = data["win_margin"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 5)
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

# https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
param_dist = {"max_depth": list(range(1,15)),
              "max_features": list(range(1,20)),
              "min_samples_leaf": list(range(1,10)),
              "min_samples_split": list(range(2,10))
              }

model = DecisionTreeRegressor()
grid = GridSearchCV(estimator = model, param_grid = param_dist)
grid.fit(x_train, y_train)

print("Tuned Decision Tree Parameters: {}".format(grid.best_estimator_))
print("Best score is {}".format(grid.best_score_))

