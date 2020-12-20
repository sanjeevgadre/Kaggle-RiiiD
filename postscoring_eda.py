#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:40:43 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#%% Helper variables
DATAPATH = './data/'

#%% Userscores from mean run

userscores = np.genfromtxt(DATAPATH + 'userscores.csv', delimiter = ',')

len(userscores[userscores[:, 8] == 0])
# 88 users remain unscored (unprocessed) after 113M transactions i.e 0.02%
mean = np.mean(userscores[:, 1:8], axis = 0)
# Mean userscores (after 3rd reward policy) - [0.02951912, 0.05758864, 0.04761834, 0.04515786, 0.2037545,  0.04537496, 0.03697]
    

np.median(userscores[:, 1:8], axis = 0)
# [0.02546064, 0.0573015 , 0.03846656, 0.03403131, 0.19177877, 0.03781714, 0.02767306]

sigma = np.std(userscores[:, 1:8], axis = 0)
# Std dev. (after 3rd reward policy )- [0.20179399, 0.21116786, 0.25834561, 0.25590359, 0.58923841, 0.21340565, 0.22842549]
    # The dispersion in userscores has come down dramatically after the reward policy change; ergo the change was appropriate

print("Coeff. of Variance for userscores -->", 
      np.std(userscores[:, 1:8], axis = 0) / np.mean(userscores[:, 1:8], axis = 0))
# 6.83604271, 3.6668316 , 5.42533904, 5.66686651, 2.89190371, 4.70315931, 6.17867076]

two_sigma_range = np.concatenate(((mean - 2*sigma).reshape(-1, 1), 
                                  (mean + 2*sigma).reshape(-1, 1)), axis = 1)

fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
COL = 1
for r in range(4):
    for c in range(2):
        if COL < 8:
            axs[r, c].hist(userscores[:, COL], bins = 100, log = True,
                           range = (two_sigma_range[COL - 1, 0], two_sigma_range[COL - 1, 1]))
            COL += 1

plt.savefig('./foo.jpg')
plt.show()

for i in range(7):
    foo = np.logical_and(userscores[:, i + 1] > two_sigma_range[i, 0], 
                         userscores[:, i + 1] < two_sigma_range[i, 1]).sum() / len(userscores)
    print('Part %i --> %.4f of total records lie within 2 sigma range' % (i+1, foo))
    
# Part 1 --> 0.9885 of total records lie within 2 sigma range
# Part 2 --> 0.9809 of total records lie within 2 sigma range
# Part 3 --> 0.9858 of total records lie within 2 sigma range
# Part 4 --> 0.9868 of total records lie within 2 sigma range
# Part 5 --> 0.9789 of total records lie within 2 sigma range
# Part 6 --> 0.9849 of total records lie within 2 sigma range
# Part 7 --> 0.9874 of total records lie within 2 sigma range
# We can safely clip the part userscores to within their respective 2 sigma range

corr_coef_ = np.corrcoef(userscores[userscores[:, 8] == 1, 1:8], rowvar = False)

# [1.        , 0.16220016, 0.22865256, 0.25533779, 0.16945078, 0.16839158, 0.14140076],
# [0.16220016, 1.        , 0.26734779, 0.24921227, 0.34266616, 0.18226742, 0.1666369 ],
# [0.22865256, 0.26734779, 1.        , 0.52356998, 0.27068343, 0.29656262, 0.32322765],
# [0.25533779, 0.24921227, 0.52356998, 1.        , 0.26399204, 0.32011388, 0.36182182],
# [0.16945078, 0.34266616, 0.27068343, 0.26399204, 1.        , 0.34576019, 0.28384395],
# [0.16839158, 0.18226742, 0.29656262, 0.32011388, 0.34576019, 1.        , 0.39792441],
# [0.14140076, 0.1666369 , 0.32322765, 0.36182182, 0.28384395, 0.39792441, 1.        ]
# There is no meaningful pairwise correlation amongst the part scores

pca_ = PCA().fit(userscores[userscores[:, 8] == 1, 1:8])
print("Variance explained by principal components --> \n", pca_.explained_variance_ratio_)
# [0.57991909, 0.15799407, 0.06727159, 0.05358795, 0.05127978, 0.04694441, 0.04300311]
# The combined variance explained by the two principal components point to likely interaction between
# the scores when building a parametric linear model.

#%% Questions
ques = np.genfromtxt(DATAPATH + 'ques.csv', delimiter = ',')

len(ques[ques[:, 2] == 0])
# 4 questions remain unasked after 145M transactions

np.mean(ques[ques[:, 2] != 0, 3])
# For the questions attempted the mean probability of answering is 0.71

# Correct Attempt Prob
fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 15))
COL = 1
for r in range(4):
    for c in range(2):
        if COL < 8:
            axs[r, c].hist(ques[ques[:, 1] == COL, 3], bins = 100)
            COL += 1

plt.savefig('./foo.jpg')
plt.show()


#%% Lectures
lecs = np.genfromtxt(DATAPATH + 'lecs.csv', delimiter = ',')

len(lecs[lecs[:, 2] == 0])
# 3 lectures remain without a single view after 145M transactions

