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
CHUNKSIZE_ = 10 * 10**6
EPOCHS = 2

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
val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val = data_preproc(val)
lab_val = np.copy(val[:, 0])
val = val[:, 1:]
val = xgb.DMatrix(val, lab_val)

# Setting up the training set for initial model
train_idx = np.random.choice(98278587, 10 * 10**6, replace = False)
train = pd.read_hdf(INPUTPATH + 'train_proc_train.h5', 'df', mode = 'r', where = pd.Index(train_idx))
train = train.sample(frac = 1)
train = data_preproc(train)
lab_train = np.copy(train[:, 0])
train = train[:, 1:]
train = xgb.DMatrix(train, lab_train)
# releasing memory
lab_train = None            
train_idx = None

# Setting up the hyperparameters
filehandle = INPUTPATH + 'tuned_params.json'
with open(filehandle, 'r') as fh:
    params = json.load(fh)
    
# Setting up the initial model   
model = xgb.train(params, train)
probs = model.predict(val)
probs = probs.reshape(-1, 2)
auc = roc_auc_score(lab_val, probs[:, 1])
print('Validation set auc: %.6f' % auc)

#%% Iterating through the training set
# Updating the parameters to enable continuing learning
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
        probs = model.predict(val)
        probs = probs.reshape(-1, 2)
        auc = roc_auc_score(lab_val, probs[:, 1])
        print('After chunk: %i, Validation set auc: %.6f' % (chunk_count, auc))
        
        chunk_count += 1