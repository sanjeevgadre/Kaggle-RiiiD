# %% [code]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:34:03 2020

@author: sanjeev

mean_run v3
"""

#%% Libraries
import pandas as pd
import numpy as np
import sys
import datetime

#%% Helper variables
inputpath = '../input/riiid-scoring/'


# max change in score that indicates steady state
epsilon = 10**(-5)
# max change in score that indicates batch steady state
epsilon_batch = 10**(-2)

# number of records in the the train dataset (known from EDA)
train_n = 101230332

# setting batch and minibatch sizes
batch_size = 1000000
minibatch_size = 1000

# iteration counts
batch_iters = 16
minibatch_iters = 1000

# timeout for Kaggle processing
time_out = 8.75 * 3600

#%% Helper functions
def get_q_reward(i):
    '''
    For the question: 
        rewards the user if answered correctly
        updates the probability of correctly answering the question
        rewards the user if prior question's explanation was viewed, and 
        updates the total views of the prior question.

    Parameters
    ----------
    i : int
        index identifier for the batch.

    Returns
    -------
    The reward earned.

    '''
    qid = batch[i, 1]
    # Get the part number and current stats of the question
    idx =  np.where(ques[:, 0] == qid)
    _, part, attempts, prob, prior_views = ques[idx].flatten()
    correct_attempts = attempts * prob
         
    # If answered correctly, calculate the reward      
    if batch[i, 3] == 1:
        correct_attempts += 1
        reward = (1. - prob) / correct_attempts
        
    else:
        incorrect_attempts = attempts - correct_attempts
        incorrect_attempts += 1
        reward = (0. - prob) / incorrect_attempts
        
    # Update the question statistics
    attempts += 1
    prob = correct_attempts / attempts
    ques[idx, 2] = attempts
    ques[idx, 3] = prob
    
    # Has the explanation for the prior question viewed?
    if batch[i, 4] == 1:
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
    lid = batch[i, 1]
    # Get the part number and current views of the lecture
    idx = np.where(lecs[:, 0] == lid)
    _, part, views = lecs[idx].flatten()
    
    # Calculate the reward
    views += 1
    reward = 1 / views
    
    # update the number of views of the explanation
    lecs[idx, 2] = views
        
    return reward, part

def update_userscores(i, reward, part, userscores_masked):
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
    userscores_masked : np.array
        Array of userscores.

    Returns
    -------
    Updated userscores.

    '''
    col_ = np.int(part)
    idx_ = np.where(userscores_masked[:, 0] == batch[i, 0])[0][0]
    userscores_masked[idx_, col_] += reward
    
    return userscores_masked

def save_data():
    '''
    Saves the processed batch data.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    '''
    np.savetxt('ques.csv', ques, delimiter = ',')
    np.savetxt('lecs.csv', lecs, delimiter = ',')
    np.savetxt('userscores.csv', userscores, delimiter = ',')
    
    return

def batch_stats(empty_minibatch_count, mean_score_beg):
    '''
    Prints the stats for the processed batch data

    Parameters
    ----------
    empty_minibatch_count : int
        Number of minibatches with no new user recorded.
    mean_score_beg : np.array
        Mean part scores at the beginning of batch processing.

    Returns
    -------
    mean_score_beg : np.array
        Mean part scores at the end of batch processing

    '''
    print('Empty minibatches -->', empty_minibatch_count)
    mean_score_end = userscores[:, 1:8].mean(axis = 0).astype(float)
    print('Part mean scores after this run -->', mean_score_end)
    mean_score_delta = mean_score_end - mean_score_beg
    mean_score_delta = mean_score_delta/mean_score_beg
    print('Change in part mean scores -->', mean_score_delta)
    mean_score_beg = mean_score_end
    
    return mean_score_beg

#%% Getting the supplementary data files
try:
    userscores = np.genfromtxt(inputpath + 'userscores.csv', delimiter = ',')
    ques = np.genfromtxt(inputpath + 'ques.csv', delimiter = ',')
    lecs = np.genfromtxt(inputpath + 'lecs.csv', delimiter = ',')
except OSError:
    print('Supplementary data files not found... exiting')
    sys.exit()
    
#%% Column names for the arrays in this program
# ques.columns -> ['question_id', 'part', 'attempts', 'correct_attempt_prob', 'prior_q_expln_views']

# lecs.columns -> ['lecture_id', 'part', 'views']

# userscores.columns -> ['user_id', 'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6', 
#                        'score_7', 'first_process_flag']

# batch.columns --> ['user_id', 'content_id', 'content_type_id', 'answered_correctly',
#                    'prior_question_had_explanation']


#%% Working with train dataset to arrive at user scores
if __name__ == '__main__':    
    mean_score_beg = userscores[:, 1:8].mean(axis = 0).astype(float)
    iterate = True
    batch_count = 0
    tic = datetime.datetime.now()
    
    while iterate and batch_count < batch_iters:
        batch_count += 1
        print('Processing batch: ', batch_count)
        batch_idx = np.random.choice(int(train_n), batch_size, replace = False)
        batch_idx = np.sort(batch_idx)
        batch = pd.read_hdf(inputpath + 'train.h5', 'df', mode = 'r', where = pd.Index(batch_idx))
        batch = batch.sample(frac = 1)
        batch.reset_index(inplace = True, drop = True)
        batch = batch.to_numpy(dtype = float)
        
        # Sanity Check
        mask = np.isin(batch[:, 0], userscores[:, 0], invert = True)
        if sum(mask) != 0:
            offending_ids = batch[mask]
            np.savetxt('offending_ids', offending_ids, delimiter = ',')
            print('Batch has a userid that is not in Userscores. Exiting')
            sys.exit()
        
        iterate_batch = True
        minibatch_count = 0
        empty_minibatch_count = 0
        
        while iterate_batch and minibatch_count < minibatch_iters:
            minibatch_count += 1
            
            # Setup a minibatch
            minibatch_idx = np.random.choice(batch_size, minibatch_size, replace = False)
            # Identify unique userids that in the minibatch
            minibatch_userids = np.unique(batch[minibatch_idx, 0])
            # Create a mask for userscores to filter minibatch userids
            minibatch_mask = np.isin(userscores[:, 0], minibatch_userids, assume_unique = True)
            # Identify the index in userscores for minibatch userids
            first_process_idx = np.where(minibatch_mask)[0]
            
            # Filter the index for userids that are being processed for the first time
            first_process_idx = [i for i in first_process_idx if userscores[i, 8] == 0]
            # For these minibatch userids, set the part scores to the current mean scores
            if len(first_process_idx) != 0:
                curr_mean_scores = userscores[:, 1:8].mean(axis = 0).astype(float)
                userscores[first_process_idx, 1:8] = curr_mean_scores
                userscores[first_process_idx, 8] = 1
            else:
                empty_minibatch_count += 1
                       
            # for each record in the minibatch
            for i in minibatch_idx:
                # Get the part number and the reward earned for the question/lecture
                if batch[i, 2] == 0:
                    reward, part = get_q_reward(i)
                else:
                    reward, part = get_l_reward(i)
                       
                # update the relevant part score for the user
                userscores[minibatch_mask] = update_userscores(i, reward, part, userscores[minibatch_mask])
                
            # Am I running out of time on Kaggle?
            if minibatch_count % 250 == 0:
                toc = datetime.datetime.now()
                if (toc-tic).total_seconds() >= time_out:
                    print('Pre-emptive exit to avoid timeout')
                    print('Batch: ', batch_count, 'Minibatch: ', minibatch_count)
                    mean_score_beg = batch_stats(empty_minibatch_count, mean_score_beg)
                    save_data()
                    sys.exit()
            
        # Print batch stats
        mean_score_beg = batch_stats(empty_minibatch_count, mean_score_beg)
        
        toc = datetime.datetime.now()
        
    save_data()
        
        
