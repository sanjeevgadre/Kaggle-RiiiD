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


'''
Chunk count:  15
Training until validation scores don't improve for 100 rounds
Early stopping, best iteration is:
[1383]	eval's auc: 0.740118
'''