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

#%% Userscores
userscores = np.genfromtxt(datapath + 'userscores.csv', delimiter = ',')

len(userscores[userscores[:, 8] == 0])
# 113 users remain unscored (unprocessed) after 1st epoch i.e 0.03%
np.mean(userscores[:, 1:8], axis = 0)
# Mean userscores - [0.03934268, 0.08991406, 0.07074137, 0.05179744, 0.26879469, 0.0540276 , 0.05278917]

# [0.06663248, 0.01301413, 0.01666397, 0.05018397, 0.03309288, 0.05533283, 0.01487584]

np.std(userscores[:, 1:8], axis = 0)
# Std dev. - [6.86006676, 17.29906613, 12.64863422, 10.86968324, 37.78130921, 15.97296181, 10.45584508])
# The scores seem very widely dispersed
np.std(userscores[:, 1:8], axis = 0) / np.mean(userscores[:, 1:8], axis = 0)
# Coeff. of var. [174.32 , 192.35, 178.74975687, 209.78956707, 140.51789314, 295.55963283, 198.01114767]
userscores = userscores[userscores[:, 8] != 0]

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
col_ = 1
for r in range(4):
    for c in range(2):
        if col_ < 8:
            axs[r, c].hist(userscores[:, col_], bins = 1000, log = True)
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