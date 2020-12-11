#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:43:58 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import glob
import random

#%% Helper variables
inputpath = './data/postscore_proc/'

#%% Merge data
fnames = glob.glob(inputpath + '*.h5')
random.shuffle(fnames)
idx_ = 0
for f in fnames:
    reader = pd.read_hdf(f, 'df', mode = 'r', iterator = True, chunksize = 10**6)
    for chunk in reader:
        # save only records that are "questions"
        qidx = chunk[chunk['content_type_id'] == 0].index
        chunk = chunk.iloc[qidx, :]
        chunk = chunk.sample(frac = 1)
        chunk.set_index(np.arange(idx_, idx_ + len(chunk), 1), inplace = True)
        idx_ += len(chunk)
        chunk.to_hdf(inputpath + 'train_proc.h5', 'df', mode = 'a', 
                     append = True, format = 'table')
        
# There are 99271300 records in the train database
