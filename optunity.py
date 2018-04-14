# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 08:38:25 2018

@author: Ithier
"""
from functools import reduce
import requests
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
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

import optunity
import optunity.metrics
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn import preprocessing



outer_cv = optunity.cross_validated(x=X, y=y, num_folds=3)

def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma, epsilon):
        pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('svr', SVR(C = C, gamma = gamma, epsilon = epsilon))])
        model = pipe.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[1, 100], gamma=[0, 50], epsilon = [0,2])
    print("optimal hyperparameters: " + str(optimal_pars))
    
    model = Pipeline([('scaler', preprocessing.StandardScaler()), ('svr', SVR(**optimal_pars))])
    tuned_model = model.fit(x_train, y_train)
    predictions = tuned_model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

# wrap with outer cross-validation
compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned(x_train, y_train, x_test, y_test))
