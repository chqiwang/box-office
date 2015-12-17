# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:16:59 2015

@author: Sophia
"""

import pickle
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor

def score(regressor,X_test,y):    
    y_pred = np.abs(regressor.predict(X_test))
    diff = np.divide(np.abs(y_pred-y),y)
    return np.mean(diff)

def train_with_search_index(X,Y,W):    
    result = []
    regressors = []
    kf = cross_validation.KFold(n=len(Y), n_folds=10, shuffle=True,random_state=None)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        W_train, W_test = W[train_index], W[test_index]
        regressor = GradientBoostingRegressor(loss='lad',n_estimators=25,learning_rate=0.1)
        regressor.fit(X_train,y_train,W_train)
        result.append(score(regressor,X_test,y_test))
        regressors.append(regressor)
    
    with open('model.pickle','w') as f:
        pickle.dump(regressors[np.argmin(result)],f)