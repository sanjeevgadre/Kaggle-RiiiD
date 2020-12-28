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
import optuna
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

#%% Objective Function
def objective(trial):
    
    # Setting up the classifier
    params = {
        'verbosity' : 0,
        'objective': 'multi:softprob',
        'num_class' : 2,
        'num_boost_round' : 1000,
        'early_stopping_rounds' : 100,
        'verbose_eval' : 250,
        'booster' : trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'tree_method' : trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'eta' : trial.suggest_loguniform('eta', 1e-3, 1),
        'gamma' : trial.suggest_loguniform('gamma', 1e-3, 1),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 1),
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 1),
        'max_depth' : trial.suggest_int('max_depth', 10, 20, 1),
        'min_child_weight' : trial.suggest_loguniform('min_child_weight', 1e-3, 100),
        'subsample' : trial.suggest_discrete_uniform('subsample', 0.1, 1, 0.1),
        'colsample_bylevel' : trial.suggest_discrete_uniform('colsample_bylevel', 0.4, 1, 0.1),
        'grow_policy' : trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    }
    if params['grow_policy'] == 'lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 10, 500, 10)
    if params['tree_method'] == 'hist':
        params['max_bins'] = trial.suggest_int('max_bins', 10, 500, 10)
    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_discrete_uniform('rate_drop', 0.1, 0.7, 0.1)
    if params['booster'] == 'gblinear':
        params['updater'] = trial.suggest_categorical('updater', ['shotgun', 'coord_descent'])
        params['feature_selector'] = trial.suggest_categorical(
                                                        'feature_selector', ['cyclic', 'shuffle']
                                                        )
    # Tune the model
    bst = xgb.train(params, train)
    if params['booster'] == 'dart':
        probs = bst.predict(val, ntree_limit = 1000)
    else:
        probs = bst.predict(val)
    probs = probs.reshape(-1, 2)
    auc = roc_auc_score(lab_val, probs[:, 1])
    
    return auc
    

#%% Data
# Setting up the validation set
val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val = data_preproc(val)
lab_val = np.copy(val[:, 0])
val = val[:, 1:]
val = xgb.DMatrix(val, lab_val)

# Setting up the training set
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

# Setting up the fixed params
tuned_params = {
        'verbosity' : 0,
        'objective': 'multi:softprob',
        'num_class' : 2,
        'num_boost_round' : 1000,
        'early_stopping_rounds' : 100,
        'verbose_eval' : 250
        }

#%% Tuning the model    
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 100)

#%% Using the tuned hyperparameters
trial = study.best_trial
for key, value in trial.params.items():
        print('  {}: {}'.format(key, value))
'''
  booster: gbtree
  tree_method: hist
  eta: 0.7643257419572588
  gamma: 0.00543886302610918
  alpha: 0.016910770780200608
  lambda: 0.09415122213411324
  max_depth: 14
  min_child_weight: 51.1561633844523
  subsample: 0.9
  colsample_bylevel: 0.8
  grow_policy: depthwise
  max_bins: 320
'''
# Updating tuned_params dictionary
tuned_params.update(study.best_params)
# Saving tuned_params for future use
filehandle = INPUTPATH + 'tuned_params.json'
with open(filehandle, 'w') as fh:
    json.dump(tuned_params, fh)
