#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:46:54 2020

@author: sanjeev
"""
import pandas as pd

datapath = './data/'
try:
    foo = pd.read_csv(datapath + 'doesnotexist.csv')
except FileNotFoundError:
    print('file not found')
