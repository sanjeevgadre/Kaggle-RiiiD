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
import lightgbm as lgb

#%% Helper variables
INPUTPATH = './data/'
CHUNKSIZE = 10 * 10**6

#%% Fitting a lgboost classifier model
# Setting up the validation set
val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val['part'] = val['part'].astype('category')
labels = val['answered_correctly']
val = val.drop(columns = 'answered_correctly')
val = lgb.Dataset(val, labels, feature_name = 'auto', categorical_feature = 'auto')

# Setting up the classifier
params = {'task' : 'train', 'objective': 'binary', 'learning_rate' : 0.01,
          'num_threads' : 2, 'force_col_wise' : True, 'max_depth' : 4,
          'min_sum_hessian_in_leaf' : 0.5, 'bagging_fraction' : 0.5,
          'first_metric_only' : True, 'metric' : 'auc', 'verbosity' : -1}
NUM_BOOST_ROUND_ = 1000
model = None

tic = dt.datetime.now()
reader = pd.read_hdf(INPUTPATH + 'train_proc_train.h5', key = 'df', mode = 'r', 
                     iterator = True, chunksize = CHUNKSIZE, stop = 1 * CHUNKSIZE)

chunk_n = 1
for chunk in reader:
    # Shuffle the chunk
    chunk = chunk.sample(frac = 1)
    chunk['part'] = chunk['part'].astype('category')
    labels = chunk['answered_correctly']
    chunk = chunk.drop(columns = 'answered_correctly')
    chunk = lgb.Dataset(chunk, labels, feature_name = 'auto', categorical_feature = 'auto')
    # Fit the model
    model = lgb.train(params, chunk, verbose_eval = False, init_model = model,
                      num_boost_round = NUM_BOOST_ROUND_,
                      early_stopping_rounds = np.int(0.1 * NUM_BOOST_ROUND_), 
                      valid_sets = [val, chunk], valid_names = ['eval', 'train'])
    print('\nChunk count: %i, Validation set auc: %f' % (chunk_n, model.best_score['eval']['auc']))
    chunk_n += 1
    
toc = dt.datetime.now()
print('Time taken: %f minutes' % ((toc-tic).total_seconds()/60))

#%%

val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val['part'] = val['part'].astype('category')
preds = model.predict(val.iloc[:, 1:])

pd.crosstab(preds, val[:, 0], normalize = True)

# Chunk count: 1, Validation set auc: 0.734444 Trg size 10*10**6
