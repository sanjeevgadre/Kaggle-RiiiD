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
def get_curr_mean_scores(userscores):
    '''
    Calculates the part-wise median scores for all users

    Returns
    -------
    curr_mean_scores : np.array
        Partwise median scores for all users.

    '''
    curr_mean_scores = np.zeros(7)
    if userscores.size == 0:
        curr_mean_scores = np.zeros(7)
    else:
        curr_mean_scores = userscores[:, 1:].mean(axis = 0)
        
    curr_mean_scores = curr_mean_scores.astype(float)
         
    return curr_mean_scores

def get_reward(i):
    '''
    For the record gets the reward and the relevant part to which the question/lecture belongs.

    Parameters
    ----------
    i : int
        index indentifier for the batch
        index identifier for the batch.

    Returns
    -------
    The reward earned and the part.

    '''
    if batch[i, 3] == 0:
        reward, part = get_q_reward(i)
    else:
        reward, part = get_l_reward(i)
       
    return reward, part

def get_q_reward(i):
    '''
    For the question, rewards the user if answered correctly. Also updates the probability of correctly answering the question. Additionally rewards the user if prior question's explanation was viewed and also updates the total views of the prior question.'

    Parameters
    ----------
    i : int
        index identifier for the batch.

    Returns
    -------
    The reward earned.

    '''
    qid = batch[i, 2]
    # Get the part number and current stats of the question
    idx =  np.where(ques[:, 0] == qid)
    _, part, attempts, prob, prior_views = ques[idx].flatten()
         
    # Calculate the reward      
    reward = np.float(batch[i, 4]) - prob
    # Update the question's statistics
    correct_attempts = attempts * prob
    if np.float(batch[i, 4]) == 1.:
        correct_attempts += 1
        
    attempts += 1
    prob = correct_attempts / attempts
    
    ques[idx, 2] = attempts
    ques[idx, 3] = prob
    
    # Has the explanation for the prior question viewed?
    if batch[i, 5] == 1:
        prior_views += 1
        reward += 1/prior_views
        ques[idx, 4] = prior_views
        
    
    return reward, part

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
    # Get the part number and current views of the lecture
    idx = np.where(lecs[:, 0] == lid)
    _, part, views = lecs[idx].flatten()
    
    # Calculate the reward
    views += 1
    reward = 1 / views
    
    # update the number of views of the explanation
    lecs[idx, 2] = views
        
    return reward, part

def update_userscores(i, reward, part, userscores):
    '''
    Updates the relevant score for the user

    Parameters
    ----------
    i : int
        Index identifier for the batch
    reward : float
        Reward earned for the question/lecture.
    part : float
        The part to which the question/lecture belongs.
    userscores : np.array
        Array of userscores.

    Returns
    -------
    Updated userscores.

    '''
    part = np.int(part)
    userid_ = np.where(userscores[:, 0] == batch[i, 1])[0][0]
    userscores[userid_, part] += reward
    
    return userscores
    
#%% Getting the supplementary data files
try:
    userscores = pd.read_csv(datapath + 'userscores.csv', header = None)
    ques = pd.read_csv(datapath + 'ques.csv', header = None)
    lecs = pd.read_csv(datapath + 'lecs.csv', header = None)
except FileNotFoundError:
    # Setting up dataframes to track scores and other statistics
    userscores = pd.DataFrame(data = None, columns = ['user_id', 'score_1', 'score_2', 'score_3', 
                                                  'score_4', 'score_5', 'score_6', 'score_7'])
    ques = pd.read_csv(datapath + 'questions.csv', usecols = ['question_id', 'part'])
    lecs = pd.read_csv(datapath + 'lectures.csv', usecols = ['lecture_id', 'part'])
    
    ques.loc[:, ['attempts', 'correct_attempt_prob', 'prior_q_expln_views']] = 0.
    lecs.loc[:, ['views']] = 0.
    
# Converting dataframes to numpy arrays
ques = ques.to_numpy(dtype = np.float32)
lecs = lecs.to_numpy(dtype = np.int32)
userscores = userscores.to_numpy(dtype = np.float32)
   
store = pd.HDFStore(datapath + 'train.h5')
train_n = store.get_storer('df').nrows
tables.file._open_files.close_all()

#%% Column names for the arrays in this program
# ques.columns -> ['question_id', 'part', 'attempts', 'correct_attempt_prob', 'prior_q_expln_views']

# lecs.columns -> ['lecture_id', 'part', 'views']

# userscores.columns -> ['user_id', 'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6', 
#                        'score_7']

# batch.columns --> ['row_id', 'user_id', 'content_id', 'content_type_id', 'answered_correctly',
#                    'prior_question_had_explanation']


#%% Working with train dataset to arrive at user scores
if __name__ == '__main__':    
    batch_size = int(p_read * train_n)
    minibatch_size = 1000
    
    iterate = True
    batch_count = 0
    
    while iterate and batch_count < 100:
        tic = datetime.datetime.now()
        
        batch_count += 1
        print('Processing batch: ', batch_count)
        batch_idx = np.random.choice(int(train_n), batch_size, replace = False)
        batch_idx = np.sort(batch_idx)
        batch = pd.read_hdf(datapath + 'train.h5', 'df', mode = 'r', where = pd.Index(batch_idx))
        batch = batch.sample(frac = 1)
        batch.reset_index(inplace = True, drop = True)
        batch = batch.to_numpy(dtype = np.float32)
        
        batch_max_reward = 0.
        iterate_batch = True
        minibatch_count = 0
        
        while iterate_batch and minibatch_count < 100:
            minibatch_count += 1
            # print('Processing minibatch: ', minibatch_count)
            
            # Setup a minibatch
            minibatch_idx = np.random.choice(batch_size, minibatch_size, replace = False)
            curr_mean_scores = get_curr_mean_scores(userscores)
            
            # Identify usersids that are in minibatch but not in userscores
            minibatch_userids = np.unique(batch[minibatch_idx, 1])
            mask = np.isin(minibatch_userids, userscores[:, 0], assume_unique = True, invert = True)
            newusers = minibatch_userids[mask].reshape(-1, 1)
            # For such userids create records in userscores
            if len(newusers) != 0:
                newusers = np.concatenate([newusers, np.array(len(newusers) * [curr_mean_scores])], 
                                          axis = 1)
                userscores = np.concatenate([userscores, newusers], axis = 0)
            
            # for each record in the minibatch
            for i in minibatch_idx:
                # Get the part number and the reward earned for the question/lecture
                reward, part = get_reward(i)
                
                # update the relevant part score for the user
                userscores = update_userscores(i, reward, part, userscores)
                           
                # Is the reward earned the maximum absolute reward for this batch?
                if np.abs(reward) > batch_max_reward:
                    batch_max_reward = np.abs(reward)
              
            # Should I stop iterating this batch?
            if batch_max_reward < epsilon_batch:
                print('stopping iteration of batch %i after %i minibatches' 
                      % (batch_count, minibatch_count))
                iterate_batch = False
        
        print('max reward for the last batch', batch_max_reward) # <-------------
                
        # Should stop iterating
        if batch_max_reward < epsilon:
            print('Scores stabilised after %i batches' % batch_count)
            iterate = False
            
        # Save the updated userscores, ques and lecs files
        np.savetxt(datapath + 'ques.csv', ques, delimiter = ',')
        np.savetxt(datapath + 'lecs.csv', lecs, delimiter = ',')
        np.savetxt(datapath + 'userscores.csv', userscores, delimiter = ',')
        
        toc = datetime.datetime.now()
        print('Time to process the batch', toc - tic)
        
# datetime.timedelta(seconds=4004, microseconds=87446)
#  datetime.timedelta(seconds=259, microseconds=175555)
