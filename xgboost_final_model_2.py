#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:08:43 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import json

#%% Helper variables
INPUTPATH = './data/'

# OneHotEncoder to encode the questions part number
part_enc = OneHotEncoder(categories = [np.arange(1, 8, 1)], dtype = 'int', sparse = False)

#%% Helper functions
def data_preproc(df):
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

#%% Data
# Setting up the validation set
dval = xgb.DMatrix('./data/train_proc_val.csv?format=csv&label_column=0#dval.cache')
lab_val = pd.read_csv('./data/train_proc_val.csv', usecols = [0])
# Setting up the training set for initial model
dtrain = xgb.DMatrix('./data/train_proc_train.csv?format=csv&label_column=0#dtrain.cache')

# Setting up the hyperparameters
params = {
    'verbosity' : 0,
    'nthread' : 2,
    'tree_method' : 'hist',
    'objective' : 'binary:logistic'
    }

'''
filehandle = INPUTPATH + 'tuned_params.json'
with open(filehandle, 'r') as fh:
    params = json.load(fh)
'''
    
# Setting up the initial model
model = xgb.train(params, dtrain)
probs = model.predict(dval)
probs = probs.reshape(-1, 2)
auc = roc_auc_score(lab_val, probs[:, 1])
print('Validation set auc: %.6f' % auc)

# 10M random training records --> Validation set auc: 0.744721
# 15M random training records --> Validation set auc: 0.745224
# 25M random training records --> Validation set auc: 0.746901 

#%% Save model for future use
model.save_model(INPUTPATH + 'xgbmodel_all.bin')

# foo = xgb.Booster(model_file = INPUTPATH + 'xgb10model.bin')

#%% Iterating through the training set
# Updating the parameters to enable continuing learning

# xgboost does not support continuing learning, certainly not for classifier models
'''
params.update({
    'process_type': 'update',
    'updater': 'refresh',
    'refresh_leaf': True,
})

for e in range(EPOCHS):
    print('Processing epoch: %i' % (e+1))
    reader = pd.read_hdf(INPUTPATH + 'train_proc_train.h5', 'df', mode = 'r',
                         iterator = True, chunksize = CHUNKSIZE_)
    chunk_count = 1
    for chunk in reader:
        chunk = chunk.sample(frac = 1)
        chunk = data_preproc(chunk)
        lab_chunk = np.copy(chunk[:, 0])
        chunk = chunk[:, 1:]
        chunk = xgb.DMatrix(chunk, lab_chunk)
        # releasing memory
        lab_chunk = None
        
        model = xgb.train(params, chunk, xgb_model = model)
        probs = model.predict(dval)
        probs = probs.reshape(-1, 2)
        auc = roc_auc_score(lab_val, probs[:, 1])
        print('After chunk: %i, Validation set auc: %.6f' % (chunk_count, auc))
        
        chunk_count += 1
'''
