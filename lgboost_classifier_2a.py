#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:08:43 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import lightgbm as lgb

#%% Helper variables
INPUTPATH = './data/'
CHUNKSIZE_ = 8 * 10**6

#%% Fitting a lgboost classifier model
# Setting up the validation set
val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val['part'] = val['part'].astype('int').astype('category')
labels = val['answered_correctly']
val = val.drop(columns = 'answered_correctly')
val = lgb.Dataset(val, labels,
                  feature_name = 'auto', categorical_feature = 'auto',
                  free_raw_data = False)

# Setting up the parameters
params = {
    'objective': 'binary',
    'metric' : 'auc',
    'verbosity' : -1,
    'num_iterations' : 1000,
    'early_stopping_rounds' : 100,
    'num_threads' : 2    
}

# Setting up the train set
reader = pd.read_hdf(INPUTPATH + 'train_proc_train.h5', 'df', mode = 'r',
                    chunksize = CHUNKSIZE_, iterator = True)
model = None
chunk_count = 1
for chunk in reader:
    print('Chunk count: ', chunk_count)
    chunk = chunk.sample(frac = 1)
    chunk['part'] = chunk['part'].astype('int').astype('category')
    labels = chunk['answered_correctly']
    chunk = chunk.drop(columns = 'answered_correctly')
    chunk = lgb.Dataset(chunk, labels,
                        feature_name = 'auto', categorical_feature = 'auto',
                        free_raw_data = False)
    model = lgb.train(params, chunk, init_model = model,
                      valid_sets = [val], valid_names = ['eval'], verbose_eval = 250)
    chunk_count += 1


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