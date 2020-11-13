#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:34:03 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import numpy as np
import tables
import datetime
import numba

#%% Helper variables
datapath = './data/'
# probability that a row will be included in the bootstrap sample
p_read = 0.001 # will read approx 100000 records

# max change in score that indicates steady state
epsilon = 10**(-5)
# max change in score that indicates batch steady state
epsilon_batch = 10**(-2)

#%% Helper functions
def get_curr_mean_scores():
    '''
    Calculates the part-wise median scores for all users

    Returns
    -------
    curr_mean_scores : np.array
        Partwise median scores for all users.

    '''
    curr_mean_scores = userscores.iloc[:, 1:].mean(skipna = True).to_numpy()
    for i in range(len(curr_mean_scores)):
        if np.isnan(curr_mean_scores[i]):
            curr_mean_scores[i] = 0
            
    return curr_mean_scores

@numba.jit
def get_reward(i):
    '''
    For the record gets the reward and the relevant part to which the question/lecture belongs.

    Parameters
    ----------
    i : int
        index identifier for the batch.

    Returns
    -------
    The reward earned and the part.

    '''
    if batch[i, 3] == 0:
        print('Its a question!!')
        #reward, part = get_q_reward(i)
        if batch[i, 5] == 1:
            print('Previous question had an explanation!!')
            #reward = reward + get_e_reward(i)     
    else:
        print('Its a lecture!!')
        #reward, part = get_l_reward(i)
       
    return #reward, part

@numba.jit
def get_q_reward(i):
    '''
    For the question, rewards the user if answered correctly. Also updates the probability of correctly answering the question.

    Parameters
    ----------
    i : int
        index identifier for the batch.

    Returns
    -------
    The reward earned.

    '''
    
    qid = batch[i, 2]
    # idx =  qid idx = ques.query('question_id == @qid').index
    
    # We use the fact that in ques question_id == index value
    part = ques[qid, 1]
    attempts = ques[qid, 2]
    prob = ques[qid, 3]
          
    reward = np.float(batch[i, 4]) - prob
      
    #update the question's attempts and correct_attempt_prob
    correct_attempts = attempts * prob
    if np.float(batch[i, 4]) == 1.:
        correct_attempts += 1
        
    attempts += 1
    prob = correct_attempts / attempts
    
    ques[qid, 2] = attempts
    ques[qid, 3] = prob
    
    return reward, part

@numba.jit    
def get_e_reward(i):
    '''
    For the question, rewards the user if explanation for the previous question was seen. Also updates the number of views of the explanation

    Parameters
    ----------
    i : int
        index identifier for the batch.

    Returns
    -------
    The reward earned.

    '''
    qid = batch[i, 2]
    # idx = ques.query('question_id == @qid').index
    
    views = ques[qid, 5]
    views += 1
    reward = 1 / views
    
    # update the number of views of the explanation
    ques[qid, 5] = views
        
    return reward

@numba.jit
def get_l_reward(i):
    '''
    For the lecture, rewards the user for the lecture view. Also updates the number of views of the lecture.

    Parameters
    ----------
    i : int
        index identifier for the batch.

    Returns
    -------
    The reward earned.

    '''
    lid = batch[i, 2]
    #idx = lecs.query('lecture_id == @lid').index
    
    part = lecs.loc[idx, 'part']
    views = lecs.loc[idx, 'views']
    views += 1
    reward = 1 / views
    
    # update the number of views of the explanation
    lecs.loc[idx, 'views'] = views
        
    return reward.iloc[0], part.iloc[0]
    
    
#%% Getting the supplementary data files
ques = pd.read_csv(datapath + 'questions.csv', usecols = ['question_id', 'part'])
lecs = pd.read_csv(datapath + 'lectures.csv', usecols = ['lecture_id', 'part'])
store = pd.HDFStore('./data/train.h5')
train_n = store.get_storer('df').nrows
tables.file._open_files.close_all()

#%% Setting up dataframes to track scores
userscores = pd.DataFrame(data = None, columns = ['user_id', 'score_1', 'score_2', 'score_3', 
                                                  'score_4', 'score_5', 'score_6', 'score_7'])
ques.loc[:, ['attempts', 'correct_attempt_prob', 'prior_q_expln_views']] = 0.
lecs.loc[:, ['views']] = 0.

#%% Get recorded data
# userscores = pd.read_csv(datapath + 'userscores.csv')
# ques = pd.read_csv(datapath + 'ques.csv')
# lecs = pd.read_csv(datapath + 'lecs.csv')

#%% Working with train dataset to arrive at user scores
batch_size = int(p_read * train_n)
minibatch_size = 1000

iterate = True
batch_count = 0

while iterate and batch_count < 1:
    batch_count += 1
    print('Processing batch: ', batch_count)
    batch_idx = np.random.choice(int(train_n), batch_size, replace = False)
    batch_idx = np.sort(batch_idx)
    batch = pd.read_hdf('./data/train.h5', 'df', mode = 'r', where = pd.Index(batch_idx))
    batch = batch.sample(frac = 1)
    batch.reset_index(inplace = True, drop = True)
    
    # batch_max_reward = 0
    iterate_batch = True
    minibatch_count = 0
    
    while iterate_batch and minibatch_count < 1:
        minibatch_count += 1
        print('Processing minibatch: ', minibatch_count)
        
        minibatch_idx = np.random.choice(batch_size, minibatch_size, replace = False)
        minibatch_max_reward = 0
        curr_mean_scores = get_curr_mean_scores()
        
        # Identify users in minibatch that are not in userscores
        minibatch_userids = pd.Series(batch.loc[minibatch_idx, 'user_id'].unique())
        newusers = minibatch_userids[~minibatch_userids.isin(userscores.user_id)]
        # Form records for newusers with part scores equal to current mean scores
        newusers = newusers.to_numpy().reshape(-1, 1)
        newusers = np.concatenate([newusers, np.array(len(newusers) * [curr_mean_scores])], 
                                  axis = 1)
        newusers = pd.DataFrame(data = newusers, columns = userscores.columns)
        # For new users append records in userscores
        userscores = userscores.append(newusers, ignore_index = True)
        
        batch = batch.to_numpy()
        for i in minibatch_idx:
            _ = get_reward(i)
            #reward, part = get_reward(i)
            
            '''
            # update the relevant part score for the user
            idx = userscores.query('user_id == @batch.user_id[@i]').index
            userscores.iloc[idx, part] += reward
            
            # Is the reward earned the maximum absolute reward for this minibatch?
            if np.abs(reward) > minibatch_max_reward:
                minibatch_max_reward = np.abs(reward)
          
        # Should I stop iterating this batch?
        if minibatch_max_reward < epsilon_batch:
            print('stopping iteration of batch %i after %i minibatches' 
                  % (batch_count, minibatch_count))
            iterate_batch = False
    
    print('batch max reward', minibatch_max_reward) # <-------------
            
    # Should stop iterating
    if minibatch_max_reward < epsilon:
        print('Scores stabilised after %i batches' % batch_count)
        iterate = False
        
    # Save the updated userscores, ques and lecs files
    ques.to_csv(datapath + 'ques.csv', index = False)
    lecs.to_csv(datapath + 'lecs.csv', index = False)
    userscores.to_csv(datapath + 'userscores.csv', index = False)
    
    '''

# datetime.timedelta(seconds=4004, microseconds=87446)