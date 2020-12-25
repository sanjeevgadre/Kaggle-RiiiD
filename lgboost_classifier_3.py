#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:08:43 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna


#%% Helper variables
INPUTPATH = './data/'

#%% Objective Function
def objective(trial):
    
    # Setting up the classifier
    params = {
        'objective': 'multiclass',
        'num_class' : 2,
        'metric' : 'auc_mu',
        'boosting' : 'gbdt',
        'num_iterations' : 1000,
        'early_stopping_round' : 100,
        'num_threads' : 2,
        'force_row_wise' : True,
        'tree_learner' : 'serial',
        'verbosity' : -1,
        'verbose_eval' : 250,
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-3, 1),
        'num_leaves' : trial.suggest_int('num_leaves', 10, 1010, 100),
        'max_depth' : trial.suggest_int('max_depth', 1, 10, 1),
        'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 10, 100, 10),
        'min_sum_hessian_in_leaf' : trial.suggest_loguniform('min_sum_hessian_in_leaf', 1e-3, 1),
        'bagging_fraction' : trial.suggest_discrete_uniform('bagging_fraction', 0.4, 1, 0.1),
        'feature_fraction' : trial.suggest_discrete_uniform('feature_fraction', 0.4, 1, 0.1),
        'lambda_l1' : trial.suggest_loguniform('lambda_l1', 1e-3, 1),
        'lambda_l2' : trial.suggest_loguniform('lambda_l2', 1e-3, 1),
        'max_bin' : trial.suggest_int('max_bin', 100, 2500),
    }
    # Tune the model
    bst = lgb.train(params, train, valid_sets = [val], valid_names = ['val'])
    probs = bst.predict(val)
    probs = probs.reshape(-1, 2)
    auc = roc_auc_score(lab_val, probs[:, 1])
    
    return auc
    

#%% Fitting a xgboost classifier model

# Setting up the validation set
val = pd.read_hdf(INPUTPATH + 'train_proc_val.h5', key = 'df', mode = 'r')
lab_val = val['answered_correctly']
val = val.drop(columns = 'answered_correctly')
val = lgb.Dataset(val, lab_val, feature_name = 'auto', categorical_feature = 'auto')

# Setting up the training set
train_idx = np.random.choice(98278587, 10 * 10**6, replace = False)
train = pd.read_hdf(INPUTPATH + 'train_proc_train.h5', 'df', mode = 'r', where = pd.Index(train_idx))
train = train.sample(frac = 1)
lab_train = train['answered_correctly']
train = train.drop(columns = 'answered_correctly')
train = lgb.Dataset(train, lab_train, feature_name = 'auto', categorical_feature = 'auto')

#%% Setting up parameter optimization study    
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 5)

trial = study.best_trial

for key, value in trial.params.items():
        print('  {}: {}'.format(key, value))
        
# Best is trial 6 with value: 0.7327721353437733 - 10 trials
