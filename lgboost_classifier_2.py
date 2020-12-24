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
    'first_metric_only' : True,
    'verbosity' : -1,
    'boosting_type' : 'gbdt',
    'num_threads' : 2
}
# Tune the model
model = lgb.train(params, train, verbose_eval = False,
                  early_stopping_rounds = 100,
                  valid_sets = [val, train], valid_names = ['eval', 'train'])

    
toc = dt.datetime.now()
print('Time taken: %f minutes' % ((toc-tic).total_seconds()/60))

#%%

val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val['part'] = val['part'].astype('category')
preds = model.predict(val.iloc[:, 1:])

pd.crosstab(preds, val[:, 0], normalize = True)

# Chunk count: 1, Validation set auc: 0.734444 Trg size 10*10**6
