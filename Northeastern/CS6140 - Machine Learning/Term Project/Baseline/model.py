#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 02:36:03 2018

@author: amal
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, mean_squared_log_error
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import math
import pandas as pd
import numpy as np
import xgboost
import pickle

def prepare_train_test(data_test):
    data = data_test.drop([
        "service_date","event_time","event_time_sec","train_id"
    ], axis=1)

    target = 'delay'
    #Define the x and y data
    X = data.drop(target, axis=1)
    
    names = list(data.drop(target, axis=1))
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns=names)   
    y = list(data[target])
    
    return (X, y)

def calculate_metrics(predict, y_test):
    
    MSE = mean_squared_error(predict, y_test)
    RMSE = math.sqrt(MSE)
    MAE = mean_absolute_error(predict, y_test)
    
    metrics = {'MEANy': MEANy,
               'SDy':SDy,
                'MSE':MSE,
               'RMSE':RMSE,
               'EVS':EVS,
               'MAE':MAE,
               'MAPE':MAPE,
               'MBE':MBE
               }
    
    metrics = pd.DataFrame(metrics, index=[0])
    return(metrics)

def ols_model(X_train, X_test, y_train, y_test):
    
    linear_model = sm.OLS(y_train, X_train)
    linear_results = linear_model.fit()
    

    
    ols_predict = linear_results.predict(X_test)
   
    metrics = calculate_metrics(ols_predict, y_test)

    return(ols_predict, metrics)
    
def XGB_model(X_train, X_test, y_train, y_test):
    
    xgb_model = xgboost.XGBRegressor(n_estimators=50, 
                                     learning_rate=0.05, 
                                     gamma=0, 
                                     subsample=1,
                                     colsample_bytree=1, 
                                     max_depth=6)
    
    xgb_results = xgb_model.fit(X_train,y_train, verbose=True)
    xgb_predict = xgb_results.predict(X_test)
    
    metrics = calculate_metrics(xgb_predict, y_test)
        
    return(xgb_predict, metrics)
    
def glm_model(X_train, X_test, y_train, y_test):
    
    gamma_model = sm.GLM( y_train, X_train, family=sm.families.Gamma())
    gamma_results = gamma_model.fit()
    
    
    glm_predict = gamma_results.predict(X_test)
    metrics = calculate_metrics(glm_predict, y_test)
    
    return(glm_predict, metrics)
