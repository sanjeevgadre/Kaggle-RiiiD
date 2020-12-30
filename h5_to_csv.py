#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:16:37 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd

#%% Convert .h5 to .csv
reader = pd.read_hdf('./data/train_proc_val.h5', 'df', mode = 'r',
                     iterator = True, chunksize = 10 * 10**6)
chunk_count = 1
for chunk in reader:
    if chunk_count == 1:
        chunk.to_csv('./data/train_proc_val.csv', mode = 'a', index = False)
        chunk_count += 1
    else:
        chunk.to_csv('./data/train_proc_val.csv', mode = 'a', index = False, header = False)
    
#%%
foo = pd.read_csv('./data/train_proc_val.csv', nrows = 100)
