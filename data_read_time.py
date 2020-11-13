#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:00:14 2020

@author: sanjeev
"""

#%% 
import pandas as pd
import numpy as np
import tables
import datetime

reader = pd.read_csv('./data/train.csv', chunksize = 100000, usecols = [0, 2, 3, 4, 7, 9], memory_map = True)

for chunk in reader:
    idx = chunk.query('@chunk.prior_question_had_explanation.isnull()').index
    chunk.loc[idx, 'prior_question_had_explanation'] = False
    chunk.prior_question_had_explanation = chunk.prior_question_had_explanation.astype(int)
    chunk.to_hdf('./data/train.h5', key = 'df', mode = 'a', 
                  append = True, format = 'table')

tables.file._open_files.close_all()
   
store = pd.HDFStore('./data/train.h5')
nrows = store.get_storer('df').nrows
tables.file._open_files.close_all()

tic = datetime.datetime.now()

r = np.random.choice(int(nrows), int(0.0001 * nrows), replace = False)
r = np.sort(r)
batch = pd.read_hdf('./data/train.h5', 'df', mode = 'r', where = pd.Index(r))

toc = datetime.datetime.now()

# datetime.timedelta(seconds=62, microseconds=630370) for 100000 random records read