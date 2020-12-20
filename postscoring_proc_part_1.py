#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:43:17 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np

#%% Helper Variables
inputpath = './data/'

#%% Userscores pre-processing
userscores = np.genfromtxt(inputpath + 'userscores.csv', delimiter = ',')

# Identify indices of users not "processed" during scoring
idx = np.where(userscores[:, 8] == 0)[0]
# For unprocessed users, assign mean part scores
userscores[idx, 1:8] = np.mean(userscores[:, 1:8], axis = 0)

# Clip part userscores to within 2-sigma range for respective part scores
mean = np.mean(userscores[:, 1:8], axis = 0)
sigma = np.std(userscores[:, 1:8], axis = 0)
two_sigma_range = np.concatenate(((mean - 2*sigma).reshape(-1, 1), 
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
    


