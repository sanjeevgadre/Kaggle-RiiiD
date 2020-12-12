#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:43:58 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import glob
import random

#%% Helper variables
inputpath = './data/postscore_proc/'

#%% Merge data
fnames = glob.glob(inputpath + '*.h5')
random.shuffle(fnames)
count = 0
for f in fnames:
    reader = pd.read_hdf(f, 'df', mode = 'r', iterator = True, chunksize = 10**6)
    for chunk in reader:
        # save only records that are "questions"
        qidx = chunk[chunk['content_type_id'] == 0].index
        chunk = chunk.iloc[qidx, :]
        chunk = chunk.sample(frac = 1)
        chunk.set_index(np.arange(count, count + len(chunk), 1), inplace = True)
        count += len(chunk)
        chunk.to_hdf(inputpath + 'train_proc.h5', 'df', mode = 'a', 
                     append = True, format = 'table')
        
# There are 99271300 records in the train database

#%% Split into test and validation sets

# Validation set that is 1% of the total train set
val_idx = np.random.choice(99271300, 992713)

# counter to set indices for validation and train set
val_count = 0
train_count = 0
# columns required for the model
cols_ = ['answered_correctly', 'prior_question_had_explanation',
       'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6',
       'score_7', 'part', 'correct_attempt_prob']

reader = pd.read_hdf(inputpath + 'train_proc.h5', 'df', mode = 'r', iterator = True, chunksize = 10**6)

for chunk in reader:
    chunk = chunk.loc[:, cols_]
    mask = np.isin(chunk.index, val_idx, assume_unique = True)
    idx_ = chunk[mask].index
    
    val_chunk = chunk.loc[idx_, cols_]
    val_chunk.set_index(np.arange(val_count, val_count + len(val_chunk), 1), inplace = True)
    val_count += len(val_chunk)
    
    train_chunk = chunk.drop(idx_).loc[:, cols_]
    train_chunk.set_index(np.arange(train_count, train_count + len(train_chunk), 1), inplace = True)
    train_count += len(train_chunk)
    
    chunk.to_hdf('./data/train_proc_.h5', 'df', mode = 'a', append = True, format = 'table')
    val_chunk.to_hdf('./data/train_proc_val.h5', 'df', mode = 'a', append = True, format = 'table')
    train_chunk.to_hdf('./data/train_proc_train.h5', 'df', mode = 'a', append = True, format = 'table')
