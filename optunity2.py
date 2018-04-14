# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 08:38:25 2018

@author: Ithier
"""
import math
import itertools
import optunity
import optunity.metrics
import sklearn.svm
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
n = diabetes.data.shape[0]

data = diabetes.data
targets = diabetes.target

outer_cv = optunity.cross_validated(x=data, y=targets, num_folds=3)

x_train,x_test,y_train,y_test = train_test_split(data,targets,test_size = 0.3,random_state = 5)


def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma, epsilon):
        model = sklearn.svm.SVR(C=C, gamma=gamma, epsilon=epsilon).fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[1, 100], gamma=[0, 50], epsilon = [0,2])
    print("optimal hyperparameters: " + str(optimal_pars))

    tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
    predictions = tuned_model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

# wrap with outer cross-validation
compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned(x_train, y_train, x_test, y_test))
