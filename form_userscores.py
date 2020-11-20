#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:43:05 2020

@author: sanjeev
"""

import pandas as pd
import numpy as np

datapath = './data/'

reader = pd.read_csv(datapath + 'train.csv', usecols = ['user_id'], 
                     chunksize = 100000, memory_map = True)
userids = []
for chunk in reader:
    userids_ = chunk.user_id.unique().tolist()
    userids = userids + userids_
    userids = list(set(userids))  
userids = np.array(userids, dtype = float)

userscores = pd.read_csv(datapath + 'userscores.csv', header = None).to_numpy(dtype = float)

mask = np.isin(userids, userscores[:, 0])
newusers = userids[mask]
newusers = newusers.reshape(-1, 1)

curr_mean_scores = np.zeros(7)
if userscores.size != 0:
    curr_mean_scores = userscores[:, 1:].mean(axis = 0)
    
curr_mean_scores = curr_mean_scores.astype(float)

newusers = np.concatenate([newusers, np.array(len(newusers) * [curr_mean_scores])], 
                          axis = 1)
userscores = np.concatenate([userscores, newusers], axis = 0)

