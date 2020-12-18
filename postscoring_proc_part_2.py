#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:04:28 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import sys
import datetime

#%% Helper variables
train_csv_path = '../input/riiid-test-answer-prediction/'
other_csv_path = '../input/riiid-scoring/'

# timeout for Kaggle processing
time_out = 8.2 * 3600

# records to read in a single batch
chunksize_ = 10**6         

# total records to read in a single run
nrows_ = 12 * chunksize_

# no. of rows to skip from top  
skiprows_ = 1 + 1 * nrows_

#%% Processing train.csv
userscores = np.genfromtxt(other_csv_path + 'userscores.csv', delimiter = ',')
ques = np.genfromtxt(other_csv_path + 'ques.csv', delimiter = ',')

# column names for train.csv 
colnames_ = ['row_id', 'timestamp', 'user_id', 'content_id', 'content_type_id', 
             'task_container_id', 'user_answer', 'answered_correctly', 
             'prior_question_elapsed_time', 'prior_question_had_explanation']

reader = pd.read_csv(train_csv_path + 'train.csv', iterator = True, memory_map = True, names = colnames_, 
                     skiprows = skiprows_, chunksize = chunksize_, nrows = nrows_)
chunk_count = 0
tic = datetime.datetime.now()

for chunk in reader:
    print('In the loop')
    chunk_count += 1
    
    # Am I running out of time on Kaggle?
    toc = datetime.datetime.now()
    if (toc-tic).total_seconds() >= time_out:
        print('Pre-emptive exit to avoid timeout')
        print('Skiprows: ', skiprows_, 'Chunk count: ', chunk_count)
        sys.exit()
    
    # eliminating nans
    chunk.loc[chunk['prior_question_had_explanation'].isna(), 'prior_question_had_explanation'] = False
    chunk['prior_question_had_explanation'] = chunk['prior_question_had_explanation'].astype('int')
    
    # add columns for 7 part userscores, part# of the question and correct_attempt_prob for the question
    chunk.loc[:, ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 
                  'score_6', 'score_7', 'part', 'correct_attempt_prob']] = 0
    colnames_ = chunk.columns
    
    # to_numpy
    chunk = chunk.to_numpy()
    
    # identify records that are questions
    qidx = np.where(chunk[:, 4] == 0)[0]
    # new features are added only for records that are questions
    for i in qidx:
        chunk[i, 10:17] = userscores[np.where(userscores[:, 0] == chunk[i, 2])[0][0], 1:8]
        chunk[i, 17:] = ques[np.where(ques[:, 0] == chunk[i, 3])[0], [1, 3]]
            
    # convert chunk to dataframe and then write to disk
    chunk = pd.DataFrame(data = chunk, columns = colnames_)
    chunk.to_hdf('train_scored.h5', key = 'df', mode = 'a', append = True, format = 'table')
    
    
        
