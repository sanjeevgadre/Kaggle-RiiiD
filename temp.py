#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:46:54 2020

@author: sanjeev
"""

foo = pd.Series(data = np.zeros(minibatch_size))
bar = pd.Series(data = np.zeros(minibatch_size))

tic = datetime.datetime.now()
foo = pd.Series(minibatch_idx).apply(lambda x: get_reward(x))
toc = datetime.datetime.now()
