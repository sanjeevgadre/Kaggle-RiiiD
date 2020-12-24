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
        'booster' : 'gbtree',
        'tree_method' : 'hist',
        'num_boost_round' : 1000,
        'early_stopping_rounds' : 100,
        'verbose_eval' : False,
        'eta' : trial.suggest_loguniform('eta', 1e-3, 1),
        'gamma' : trial.suggest_loguniform('gamma', 1e-3, 1),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 1),
        'max_depth' : trial.suggest_int('max_depth', 1, 6, 1),
        'min_child_weight' : trial.suggest_loguniform('min_child_weight', 1e-3, 1),
        'subsample' : trial.suggest_discrete_uniform('subsample', 0.4, 1, 0.1),
        'grow_policy' : trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }
    # Tune the model
    bst = xgb.train(params, train)
    probs = bst.predict(val)
    probs = probs.reshape(-1, 2)
    auc = roc_auc_score(lab_val, probs[:, 1])
    
    return auc
    

#%% Fitting a xgboost classifier model

# Setting up the validation set
val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
val = data_preproc(val)
lab_val = np.copy(val[:, 0])
val = val[:, 1:]
val = xgb.DMatrix(val, lab_val)

# Setting up the training set
train_idx = np.random.choice(98278587, 25 * 10**6, replace = False)
train = pd.read_hdf(INPUTPATH + 'train_proc_train.h5', 'df', mode = 'r', where = pd.Index(train_idx))
train = train.sample(frac = 1)
train = data_preproc(train)
lab_train = np.copy(train[:, 0])
train = train[:, 1:]
train = xgb.DMatrix(train, lab_train)
    
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 10)

trial = study.best_trial

for key, value in trial.params.items():
        print('  {}: {}'.format(key, value))
        
# Best is trial 6 with value: 0.7327721353437733 - 10 trials
