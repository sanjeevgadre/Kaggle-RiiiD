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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score


#%% Helper variables
inputpath = './data/'
chunksize_ = 10**6
epochs_to_run = 3

# Defining the classifier
clf = SGDClassifier(loss = 'log', penalty = 'l1', warm_start = True)
# OneHotEncoder to transform the part number to which the question belongs
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

tic = dt.datetime.now()
epoch = 0
while epoch < epochs_to_run:
    reader = pd.read_hdf(inputpath + 'train_proc_train.h5', key = 'df', mode = 'r', 
                         iterator = True, chunksize = chunksize_)
    
    for chunk in reader:
        # Shuffle the chunk
        chunk = chunk.sample(frac = 1)
        # Pre-process the chunk
        chunk = model_preproc(chunk)
        # Fit the model
        model = clf.partial_fit(chunk[:, 1:], chunk[:, 0], classes = np.array([0, 1]))
    
    # Making predictions for the validation set
    probs = model.predict_proba(val[:, 1:])
    # Calculate the ROC-AUC
    val_auc = roc_auc_score(val[:, 0], probs[:, 1])
    print('Validation set area under the ROC curve after %i epochs: %f' % (epoch + 1, val_auc))
    epoch += 1

toc = dt.datetime.now()

preds = model.predict(val[:, 1:])

pd.crosstab(preds, val[:, 0], normalize = True)

# PENALTY = l2
# Validation set area under the ROC curve after 1 epochs: 0.728667
# Validation set area under the ROC curve after 2 epochs: 0.728711
# Validation set area under the ROC curve after 3 epochs: 0.728724
# Time in seconds - 2270.553519

# Cross Tab
# col_0       0.0       1.0
# row_0                    
# 0      0.121569  0.072991
# 1      0.221798  0.583642

# PENALTY = l1
# Validation set area under the ROC curve after 1 epochs: 0.728656
# Validation set area under the ROC curve after 2 epochs: 0.728683
# Validation set area under the ROC curve after 3 epochs: 0.728702

# Cross Tab
# col_0       0.0       1.0
# row_0                    
# 0      0.123703  0.075190
# 1      0.219664  0.581442

#%% Fitting a modified Huber loss function
clf = SGDClassifier(loss = 'modified_huber', warm_start = True)

val = pd.read_hdf(inputpath + 'train_proc_val.h5', key = 'df', mode = 'r')
val = model_preproc(val)

tic = dt.datetime.now()
epoch = 0
while epoch < epochs_to_run:
    reader = pd.read_hdf(inputpath + 'train_proc_train.h5', key = 'df', mode = 'r', 
                         iterator = True, chunksize = chunksize_)
    
    for chunk in reader:
        # Shuffle the chunk
        chunk = chunk.sample(frac = 1)
        # Pre-process the chunk
        chunk = model_preproc(chunk)
        # Fit the model
        model = clf.partial_fit(chunk[:, 1:], chunk[:, 0], classes = np.array([0, 1]))
    
    # Making predictions for the validation set
    probs = model.predict_proba(val[:, 1:])
    # Calculate the ROC-AUC
    val_auc = roc_auc_score(val[:, 0], probs[:, 1])
    print('Validation set area under the ROC curve after %i epochs: %f' % (epoch + 1, val_auc))
    epoch += 1

toc = dt.datetime.now()

preds = model.predict(val[:, 1:])

pd.crosstab(preds, val[:, 0], normalize = True)

# PENALTY = l2
# Validation set area under the ROC curve after 1 epochs: 0.728513
# Validation set area under the ROC curve after 2 epochs: 0.728648
# Validation set area under the ROC curve after 3 epochs: 0.728576
# Time in seconds - 1084.938777
# Crosstab
# col_0       0.0       1.0
# row_0                    
# 0      0.122766  0.074458
# 1      0.220602  0.582174
