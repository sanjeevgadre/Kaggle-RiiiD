#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:00:14 2020

@author: sanjeev
"""

#%% 
import numpy as np
import pandas as pd
import sys

datapath = './data/'

userids_disk = np.genfromtxt(datapath + 'userscores.csv', delimiter = ',')
userids_disk = userids_disk[:, 0]

reader = pd.read_hdf(datapath + 'train.h5', columns = ['user_id'], chunksize = 10000000, iterator = True)

for chunk in reader:
    userids_hdf = chunk.user_id.to_numpy()
    mask = np.isin(userids_hdf, userids_disk, invert = True)
    if sum(mask) != 0:
        offending_ids = userids_hdf[mask]
        np.savetxt(datapath + 'offending_ids', offending_ids, delimiter = ',')
        print('Batch has a userid that is not in Userscores. Exiting')
        sys.exit()
    
    
    '''userids_hdf = userids_hdf + userids_
    userids_hdf = list(set(userids_hdf))
    
userids_hdf = np.array(userids_hdf, dtype = float).reshape(-1,1)

mask = np.isin(userids_csv, userids_hdf, invert = True)
## userids in csv and hdf are the same

userids_userscores = userscores[:, 0]
mask = np.isin(userids_userscores, userids_hdf, invert = True)
## one missing userids_userscores in userids_hdf
mask = np.isin(userids_hdf, userids_userscores, invert = True)

## one missing userids_userscores in userids_csv
mask = np.isin(userids_csv, userids_userscores, invert = True)
mask = np.isin(userids_userscores, userids_csv, invert = True)

# There is some sort of switch for one userid - 1238495430 is in userscores but not in csv
# and 1238495940 is in csv but not in userscores

# Recreating the usrescores file
reader = pd.read_csv(datapath + 'train.csv', usecols = ['user_id'], chunksize = 100000, memory_map = True)

userids = []
for chunk in reader:
    userids_ = chunk.user_id.unique().tolist()
    userids = userids + userids_
    userids = list(set(userids))
    
userids = np.array(userids, dtype = float).reshape(-1,1)
userscores_from_csv = np.concatenate([userids, np.array(len(userids) * [np.zeros(8)])], 
                            axis = 1)

userscores_from_csv = userscores_from_csv[:, 0]
mask = np.isin(userids_disk, userscores_from_csv, invert = True)

mask = np.isin(userids_disk, userids_csv, invert = True)
mask = np.isin(userids_disk, userids_hdf, invert = True)'''
