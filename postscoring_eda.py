#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:40:43 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Helper variables


#%% Userscores from mean run
datapath = './kaggle/mean_run/'
userscores = np.genfromtxt(datapath + 'userscores.csv', delimiter = ',')

len(userscores[userscores[:, 8] == 0])
# 101 users remain unscored (unprocessed) after 1st epoch i.e 0.03%
np.mean(userscores[:, 1:8], axis = 0)
# Mean userscores (before reward poilcy change) - [0.03934268, 0.08991406, 0.07074137, 0.05179744, 0.26879469, 0.0540276 , 0.05278917]
    #  this alerted change in reward policy -- > [0.06663248, 0.01301413, 0.01666397, 0.05018397, 0.03309288, 0.05533283, 0.01487584]
# Mean userscores (after reward policy change) - [0.04868054, 0.09712866, 0.07744997, 0.07086877, 0.3167342 , 0.07259032, 0.05839419]
    # After reward policy change the mean rewards have gone up and this probably because the negative element in the reward policy was removed

np.median(userscores[:, 1:8], axis = 0)
# [0.02546064, 0.0573015 , 0.03846656, 0.03403131, 0.19177877, 0.03781714, 0.02767306]

np.std(userscores[:, 1:8], axis = 0)
# Std dev. (before reward policy change)- [6.86006676, 17.29906613, 12.64863422, 10.86968324, 37.78130921, 15.97296181, 10.45584508])
# Std dev. (after reward policy change)- [0.25754518, 0.2705686 , 0.32062192, 0.32267703, 0.77888801, 0.26850036, 0.28461705]
    # The dispersion in userscores has come down dramatically after the reward policy change; ergo the change was appropriate

np.std(userscores[:, 1:8], axis = 0) / np.mean(userscores[:, 1:8], axis = 0)
# Coeff. of var. (before reward policy change) - [174.32 , 192.35, 178.75, 209.78956707, 140.51789314, 295.55963283, 198.01114767]
# Coeff. of var. (after reward policy change) - [5.29, 2.79, 4.13972965, 4.5531628 , 2.4591219, 3.69884544, 4.87406438]

userscores = userscores[userscores[:, 8] != 0]

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(userscores[:, col_], bins = 100, log = True)
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(userscores[:, col_], bins = 100, log = True, cumulative = True, density = True)
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

#%% Userscores from median run
datapath = './kaggle/median_run/'
userscores = np.genfromtxt(datapath + 'userscores.csv', delimiter = ',')

len(userscores[userscores[:, 8] == 0])
# 105 users remain unscored (unprocessed) after 1st epoch i.e 0.03%
np.mean(userscores[:, 1:8], axis = 0)
# Mean userscores [0.027644  , 0.05299577, 0.04430043, 0.04097484, 0.17795478, 0.04012942, 0.03366916]
    # Using current_median_scores to boot new users reduces the mean scores for the users by ~40%

np.median(userscores[:, 1:8], axis = 0)
# [1.93309102e-06, 1.16364404e-03, 0.00000000e+00, 0.00000000e+00, 8.65756454e-03, 0.00000000e+00, 0.00000000e+00]
    
np.std(userscores[:, 1:8], axis = 0)
# Std dev. - [0.26223202, 0.2730808 , 0.31885302, 0.32461032, 0.79725281, 0.26997346, 0.28284393] 
    # The std. dev. in userscores is approximately the same across median and mean runs

np.std(userscores[:, 1:8], axis = 0) / np.mean(userscores[:, 1:8], axis = 0)
# Coeff. of var. - [9.48603631, 5.15287958, 7.1975147 , 7.92218589, 4.48008659, 6.72757004, 8.40068369]
    # Coeff of var is almost twice for median run as compared to mean_run

userscores = userscores[userscores[:, 8] != 0]

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(userscores[:, col_], bins = 100, log = True)
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

#%% Questions
ques = np.genfromtxt(datapath + 'ques.csv', delimiter = ',')

len(ques[ques[:, 2] == 0])
# 5 questions remain unasked after 1st epoch i.e. 0.0004%

np.mean(ques[ques[:, 2] != 0, 3])
# For the questions attempted the mean probability of answering is 0.71

# Correct Attempt Prob
fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(ques[ques[:, 1] == col_, 3], bins = 100)
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

# Attempts
fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(ques[ques[:, 1] == col_, 2], bins = 100, range = (0, 5000), log = True)
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

#%% Lectures
lecs = np.genfromtxt(datapath + 'lecs.csv', delimiter = ',')

len(lecs[lecs[:, 2] == 0])
# 4 lectures remain without a single view after 1st epoch i.e. 1%
np.mean(lecs[lecs[:, 2] != 0, 2])
# viewed lectures get mean views of 4765

# Views
fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(lecs[lecs[:, 1] == col_, 2], bins = 10)
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

#%% Some stuff around Pareto distribution
x = userscores[:, 5]
x = x + 1
x_m = np.min(x)
alpha = len(x)/(np.sum(np.log(x/x_m)))
# 4.42

mean = alpha*x_m/(alpha - 1)
# 1.293 (remember the disti was rightshifted by 1)
sd = np.sqrt((x_m**2 * alpha)/((alpha - 1)**2 * (alpha - 2)))
# 0.396