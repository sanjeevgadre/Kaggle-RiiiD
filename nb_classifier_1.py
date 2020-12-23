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
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import roc_auc_score

#%% Helper variables
inputpath = './data/'
chunksize_ = 10**6
epochs_to_run = 2

# OneHotEncoder to encode the questions part number
part_enc = OneHotEncoder(categories = [np.arange(1, 8, 1)], dtype = 'int', sparse = False)

#%% Helper functions
def model_preproc(df):
    '''
    Preprocess the dataframe to fit/predict using SGDClassifier

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

#%% Fitting a logistic regression model
# Setting up the validation set
val = pd.read_hdf(inputpath + 'train_proc_val.h5', key = 'df', mode = 'r')
val = model_preproc(val)

# BernoulliNB classifier for categorical variables
b_clf = BernoulliNB()
# GaussianNB classifier for continous variables
g_clf = GaussianNB()

reader = pd.read_hdf(inputpath + 'train_proc_train.h5', key = 'df', mode = 'r', 
                     iterator = True, chunksize = chunksize_)
for chunk in reader:
    # Shuffle the chunk
    chunk = chunk.sample(frac = 1)
    # Pre-process the chunk
    chunk = model_preproc(chunk)
    # Fit the BernoulliNB classifier
    b_clf.partial_fit(chunk[:, 10:], chunk[:, 0], classes = np.array([0, 1]))
    # Fit the GaissianNB classifier
    g_clf.partial_fit(chunk[:, 1:10], chunk[:, 0], classes = np.array([0, 1]))

# Making predictions for the validation set
b_probs = b_clf.predict_proba(val[:, 10:])
g_probs = g_clf.predict_proba(val[:, 1:10])
# Combining the probabolities from the two NB classifiers
# Multiplying individual classifier class probabilities and normalizing using class priors
probs = np.divide(np.multiply(b_probs, g_probs), g_clf.class_prior_)

# Calculate the ROC-AUC
val_auc = roc_auc_score(val[:, 0], probs[:, 1])
print('Validation set area under the ROC curve: %f' % val_auc)

# Validation set area under the ROC curve: 0.683644

