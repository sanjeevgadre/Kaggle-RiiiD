#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:19:42 2020

@author: sanjeev
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('scores_cython_1.pyx', language_level = '3', annotate = True), 
      include_dirs = [numpy.get_include()])