#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:17:55 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import numpy as np

#%% Helper variables
inputpath = './data/'
batch_size = 1000
train_n = 101230332

#%% Processed Train set
'''
1. Read the relevant columns from the existing train dataset
2. Given a user-id, add userscores in new columns
'''
userscores = np.genfromtxt('./kaggle/median_run/userscores.csv', delimiter = ',')
ques = np.genfromtxt('./kaggle/median_run/ques.csv', delimiter = ',')

batch = pd.read_hdf(inputpath + 'train.h5', 'df', mode = 'r', start = 0, stop = 1000).to_numpy()
# Add 9 columns - 7 columns for the 7-part userscores for the userid in the record, 1 column for part to which the question in the record belongs to and 1 column for the overall probability of answering the question correctly
batch = np.concatenate((batch, np.zeros((batch.shape[0], 9), dtype = float)), axis = 1)

q_idx = np.where(batch[:, 2] == 0)[0]
for i in q_idx[:1]:
    idx_ = np.where(userscores[:, 0] == batch[i, 0])[0]
    batch[i, 5:12] = userscores[idx_, 1:8]
    idx_ = np.where(ques[:, 0] == batch[i, 1])[0]
    batch[i, 12:] = ques[idx_, [1,3]]

# Drop the non-necessary columns from the batch    
batch = batch[:, 2:]
