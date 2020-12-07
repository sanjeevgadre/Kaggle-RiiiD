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
datapath = './data/'

#%% Userscores from mean run

userscores = np.genfromtxt(datapath + 'userscores.csv', delimiter = ',')

len(userscores[userscores[:, 8] == 0])
# 88 users remain unscored (unprocessed) after 113M transactions i.e 0.02%
mean = np.mean(userscores[:, 1:8], axis = 0)
# Mean userscores (before reward poilcy change) - [0.03934268, 0.08991406, 0.07074137, 0.05179744, 0.26879469, 0.0540276 , 0.05278917]
    #  this alerted change in reward policy -- > [0.06663248, 0.01301413, 0.01666397, 0.05018397, 0.03309288, 0.05533283, 0.01487584]
# Mean userscores (after reward policy change) - [0.04868054, 0.09712866, 0.07744997, 0.07086877, 0.3167342 , 0.07259032, 0.05839419]
    # After reward policy change the mean rewards have gone up and this probably because the negative element in the reward policy was removed

# Mean userscores (after 3rd reward policy) - [0.02951912, 0.05758864, 0.04761834, 0.04515786, 0.2037545,  0.04537496, 0.03697]
    

np.median(userscores[:, 1:8], axis = 0)
# [0.02546064, 0.0573015 , 0.03846656, 0.03403131, 0.19177877, 0.03781714, 0.02767306]

sigma = np.std(userscores[:, 1:8], axis = 0)
# Std dev. (before reward policy change)- [6.86006676, 17.29906613, 12.64863422, 10.86968324, 37.78130921, 15.97296181, 10.45584508])
# Std dev. (after reward policy change)- [0.25754518, 0.2705686 , 0.32062192, 0.32267703, 0.77888801, 0.26850036, 0.28461705]
    # The dispersion in userscores has come down dramatically after the reward policy change; ergo the change was appropriate
    
# Std dev. (after 3rd reward policy )- [0.20179399, 0.21116786, 0.25834561, 0.25590359, 0.58923841, 0.21340565, 0.22842549]
    # The dispersion in userscores has come down dramatically after the reward policy change; ergo the change was appropriate

np.std(userscores[:, 1:8], axis = 0) / np.mean(userscores[:, 1:8], axis = 0)
# Coeff. of var. (before reward policy change) - [174.32 , 192.35, 178.75, 209.78956707, 140.51789314, 295.55963283, 198.01114767]
# Coeff. of var. (after reward policy change) - [5.29, 2.79, 4.13972965, 4.5531628 , 2.4591219, 3.69884544, 4.87406438]

# Coeff. of var. (after 3rd reward policy) - 6.83604271, 3.6668316 , 5.42533904, 5.66686651, 2.89190371, 4.70315931, 6.17867076]

userscores = userscores[userscores[:, 8] != 0]

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(userscores[:, col_], bins = 100, log = True,
                           range = (mean[col_ - 1] - 2*sigma[col_ - 1], mean[col_ - 1] + 2*sigma[col_ - 1]))
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(userscores[:, col_], bins = 100, log = True, cumulative = True, density = True, 
                           range = (mean[col_ - 1] - 2*sigma[col_ - 1], mean[col_ - 1] + 2*sigma[col_ - 1]))
            col_ += 1

plt.savefig('./foo.jpg')
plt.show()

two_sigma_range_ = np.concatenate(((mean - 2*sigma).reshape(-1, 1), (mean + 2*sigma).reshape(-1, 1)), axis = 1)

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