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
import optuna.integration.lightgbm as lgb

#%% Helper variables
INPUTPATH = './data/'

#%% Fitting a lgboost classifier model
tic = dt.datetime.now()
# Setting up the validation set
val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val['part'] = val['part'].astype('category')
labels = val['answered_correctly']
val = val.drop(columns = 'answered_correctly')
val = lgb.Dataset(val, labels, feature_name = 'auto', categorical_feature = 'auto')

# Setting up the train set
train_idx = np.random.choice(98278587, 10**6, replace = False)
train = pd.read_hdf(INPUTPATH + 'train_proc_train.h5', 'df', mode = 'r', where = pd.Index(train_idx))
train = train.sample(frac = 1)
train['part'] = train['part'].astype('category')
labels = train['answered_correctly']
train = train.drop(columns = 'answered_correctly')
train = lgb.Dataset(train, labels, feature_name = 'auto', categorical_feature = 'auto')

# Setting up the classifier
params = {
    'objective': 'binary',
    'metric' : 'auc',
    'verbosity' : -1,
    'early_stopping_rounds' : 100,
    'num_threads' : 2    
}
# Tune the model
model = lgb.train(params, train, valid_sets = [val], valid_names = ['eval'],
                  verbose_eval = 250)
    
toc = dt.datetime.now()
print('Time taken: %f minutes' % ((toc-tic).total_seconds()/60))

#%%

best_params = model.params

for key, value in best_params.items():
       print("    {}: {}".format(key, value))

# Time Taken
# 297.757965 minutes

# Best Val Score
# 'auc', 0.7424837025478946

# Best Params
# objective: binary
# metric: auc
# verbosity: -1
# num_threads: 2
# feature_pre_filter: False
# lambda_l1: 0.00024028233686583582
# lambda_l2: 6.228955556730153
# num_leaves: 136
# feature_fraction: 0.5
# bagging_fraction: 0.9324985326382863
# bagging_freq: 1
# min_child_samples: 20
# num_iterations: 1000
# early_stopping_round: 100
# categorical_column: [8]