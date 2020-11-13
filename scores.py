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
    qid = batch.content_id[i]
    idx = ques.query('question_id == @qid').index
  
    part = ques.loc[idx, 'part']
    prob = ques.loc[idx, 'correct_attempt_prob']
    attempts = ques.loc[idx, 'attempts']
          
    reward = np.float(batch.answered_correctly[i]) - prob
      
    #update the question's attempts and correct_attempt_prob
    correct_attempts = attempts * prob
    if batch.answered_correctly[i]:
        correct_attempts += 1
        
    attempts += 1
    prob = correct_attempts / attempts
        
    ques.loc[idx, 'correct_attempt_prob'] = prob
    ques.loc[idx, 'attempts'] = attempts
    
    return reward.iloc[0], part.iloc[0]
    
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
    qid = batch.content_id[i]
    idx = ques.query('question_id == @qid').index
    
    views = ques.loc[idx, 'prior_q_expln_views']
    views += 1
    reward = 1 / views
    
    # update the number of views of the explanation
    ques.loc[idx, 'prior_q_expln_views'] = views
        
    return reward.iloc[0]

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
    lid = batch.content_id[i]
    idx = lecs.query('lecture_id == @lid').index
    
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

tic = datetime.datetime.now() # <-------------

while iterate and batch_count < 10:
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
    while iterate_batch and minibatch_count < 150:
        minibatch_count += 1
        print('Processing minibatch: ', minibatch_count)
        minibatch_idx = np.random.choice(int(len(batch)), minibatch_size, replace = False)  
        minibatch_max_reward = 0
        
        curr_mean_scores = get_curr_mean_scores()
        for i in minibatch_idx:
            # Get the current user scores
            idx = userscores.query('user_id == @batch.user_id[@i]').index
            if idx.shape[0] == 0: 
                # create a new user record in userscores
                idx = len(userscores)
                userscores.loc[idx, 'user_id'] = batch.user_id[i]
                userscores.iloc[idx, 1:] = curr_mean_scores

            if batch.content_type_id[i] == 0:
                reward, _part = get_q_reward(i)
                if batch.prior_question_had_explanation[i] == 1:
                   reward =reward + get_e_reward(i)     
            else:
               reward, _part = get_l_reward(i)
                      
            # update the relevant part score for the user
            userscores.iloc[idx, _part] = reward
            
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
    
toc = datetime.datetime.now() # <-------------

    # datetime.timedelta(seconds=241, microseconds=645158) for 10000
    # Processing 100,000 random records got 14% of all users in userscores
    # Processing 200,000 random records got 23% of all users in userscores

#%% Saving the current scores
ques.to_csv(datapath + 'ques.csv', index = False)  # 5 columns
lecs.to_csv(datapath + 'lecs.csv', index = False)  # 3 columns
userscores.to_csv(datapath + 'userscores.csv', index = False) # 8 columns

