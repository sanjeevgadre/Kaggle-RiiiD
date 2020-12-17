#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:08:43 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb


#%% Helper variables
inputpath = './data/'
chunksize_ = 10**6
epochs_to_run = 1

# OneHotEncoder to transform the question's part number in data
part_enc = OneHotEncoder(categories = [np.arange(1, 8, 1)], dtype = 'int', sparse = False)

#%% Helper functions
def model_preproc(df):
    '''
    Preprocess the dataframe to fit/predict using XGBclassifier

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : numpy.ndarray

    '''
    encoded_part = part_enc.fit_transform(df['part'].to_numpy().reshape(-1, 1))
    df = df.drop(columns = 'part').to_numpy()
    df = np.concatenate((df, encoded_part), axis = 1)
    
    return df


#%% Fitting a xgboost classifier model

# Setting up the validation set
val = pd.read_hdf(inputpath + 'train_proc_val.h5', key = 'df', mode = 'r')
val = model_preproc(val)
labels = np.copy(val[:, 0])
val = val[:, 1:]
val = xgb.DMatrix(val, labels)

# Calculate class weights for use in classifier
# class_wts_ = compute_class_weight('balanced', classes = np.array([0, 1]), y = val[:, 0])

# Setting up the classifier
model = None
params = {'objective': 'binary:logistic', 'eval_metric' : 'auc', 
          'max_depth': 4, 'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 0.5}
evals_result_ = {}
num_boost_round_ = 1000

tic = dt.datetime.now()
epoch = 0
while epoch < epochs_to_run:
    reader = pd.read_hdf(inputpath + 'train_proc_train.h5', key = 'df', mode = 'r', 
                         iterator = True, chunksize = chunksize_, stop = 4*chunksize_)
    
    for chunk in reader:
        # Shuffle the chunk
        chunk = chunk.sample(frac = 1)
        # Pre-process the chunk
        chunk = model_preproc(chunk)
        labels = np.copy(chunk[:, 0])
        chunk = chunk[:, 1:]
        chunk = xgb.DMatrix(chunk, labels)
        # Fit the model
        model = xgb.train(params, chunk, num_boost_round = num_boost_round_,  
                          early_stopping_rounds = 0.1 * num_boost_round_, 
                          evals = [(chunk, 'train'), (val, 'eval')], evals_result = evals_result_, 
                          verbose_eval = False, xgb_model = model)
        print('Validation set auc --> %f' % model.best_score)
    
    epoch += 1

toc = dt.datetime.now()

# preds = model.predict(val[:, 1:])

# pd.crosstab(preds, val[:, 0], normalize = True)

# PENALTY = l2
# Validation set area under the ROC curve after 1 epochs: 0.728688
# Validation set area under the ROC curve after 2 epochs: 0.728692
# Validation set area under the ROC curve after 3 epochs: 0.728703
# Time in seconds - 2270.553519

# Cross Tab
# col_0       0.0       1.0
# row_0                    
# 0      0.218628  0.202718
# 1      0.124740  0.453915

