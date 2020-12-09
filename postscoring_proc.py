#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:43:17 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import datetime
import sys

#%% Helper Variables
inputpath = './data/'
time_out = 8.5 * 3600

#%% Userscores pre-processing
userscores = np.genfromtxt(inputpath + 'userscores.csv', delimiter = ',')

# Identify indices of users not "processed" during scoring
idx = np.where(userscores[:, 8] == 0)[0]
# For unprocessed users, assign mean part scores
userscores[idx, 1:8] = np.mean(userscores[:, 1:8], axis = 0)

# Clip part userscores to within 2-sigma range for respective part scores
mean = np.mean(userscores[:, 1:8], axis = 0)
sigma = np.std(userscores[:, 1:8], axis = 0)
two_sigma_range = two_sigma_range = np.concatenate(((mean - 2*sigma).reshape(-1, 1), 
                                                    (mean + 2*sigma).reshape(-1, 1)), axis = 1)
for i in range(7):
    userscores[:, i + 1] = np.clip(userscores[:, i + 1], two_sigma_range[i, 0], two_sigma_range[i, 1])
    
# Write to disk
np.savetxt(inputpath + 'userscores.csv', userscores, delimiter = ',')
    
#%% Questions pre-processing
ques = np.genfromtxt(inputpath + 'ques.csv', delimiter = ',')

# Indentify indices of questions not "processed" during scoring
idx = np.where(ques[:, 2] == 0)[0]
# For unprocessed questions, assign mean correct_attempt_prob for the part the questions belongs to
for i in idx:
    part = ques[i, 1]
    ques[i, 3] = np.mean(ques[ques[:, 1] == part, 3])
    
# Write to disk
np.savetxt(inputpath + 'ques.csv', ques, delimiter = ',')
    
#%% Train dataset pre-processing / feature engineering
reader = pd.read_csv(inputpath + 'train.csv', iterator = True, nrows = 1000,
                     chunksize = 1000, memory_map = True)
tic = datetime.datetime.now()
chunk_count = 0

for chunk in reader:
    chunk_count += 1
    # eliminating nans
    chunk.loc[chunk['prior_question_had_explanation'].isna(), 'prior_question_had_explanation'] = False
    chunk['prior_question_had_explanation'] = chunk['prior_question_had_explanation'].astype('int')
    
    # add columns for 7 part userscores, part# of the question and correct_attempt_prob for the question
    chunk.loc[:, ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 
                  'score_6', 'score_7', 'part', 'correct_attempt_prob']] = 0
    colnames_ = chunk.columns
    
    # to_numpy
    chunk = chunk.to_numpy()
    
    # populate new feature values
    for i in range(len(chunk)):
        # new features are added only for records that are questions
        if chunk[i, 4] == 0:
            chunk[i, 10:17] = userscores[np.where(userscores[:, 0] == chunk[i, 2])[0][0], 1:8]
            chunk[i, 17:] = ques[np.where(ques[:, 0] == chunk[i, 3])[0], [1, 3]]
            
    # convert chunk to dataframe and then write to disk
    chunk = pd.DataFrame(data = chunk, columns = colnames_)
    chunk.to_hdf(inputpath + 'train_scored.h5', key = 'df', mode = 'a', append = True, 
                 format = 'table')
    
    # Am I running out of time on Kaggle
    toc = datetime.datetime.now()
    if (toc-tic).total_seconds() >= time_out:
        print('Pre-emptive exit after saving chunk number: ', chunk_count)
        sys.exit()
