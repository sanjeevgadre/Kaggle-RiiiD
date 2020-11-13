#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:44:29 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import numpy as np
import tables

#%% Helper variables
datapath = './data/'

#%% Questions
ques = pd.read_csv(datapath + 'questions.csv')

ques.shape
'''(13523, 5)'''

ques.head()
'''A single question has multiple tags associated with it'''

ques.columns
'''
ques columns: 
    question_id: foreign key for the train/test content_id column, when the content type is question (0).
    bundle_id: code for which questions are served together.
    correct_answer: the answer to the question. Can be compared with the train user_answer column to check if the user was right.
    part: the relevant section of the TOEIC test.
    tags: one or more detailed tag codes for the question. The meaning of the tags will 39not be provided.
'''

ques.nunique(axis = 0)
'''
unique counts in each column:
    question_id       13523
    bundle_id          9765
    correct_answer        4
    part                  7
    tags               1519
'''

s = ques.groupby('bundle_id')['question_id'].count()
s.sum()
s.value_counts(normalize = True, sort = True)
'''
Since no questions have missing bundle_id and s.sum() == len(ques.question_id), we can surmise that each question is included in just a single bundle.

Percentage of bundles with number of questions in them:
    # ques
    1    0.834716
    3    0.107834
    4    0.038812
    5    0.011367
    2    0.007271
    
A large majority of bundles have a single question in them.
'''

s = ques.groupby('part')['question_id'].count()
s.sum()
s/s.sum()
'''
Since no questions have missing part and s.sum() == len(ques.question_id), we can surmise that each question is included in just a single part.

Percentage of questions in different parts:
    part
    1    0.073357
    2    0.121793
    3    0.115507
    4    0.106411
    5    0.407528
    6    0.089625
    7    0.085780
'''

s = ques.groupby(['part'])['bundle_id'].nunique()
s.sum()
s/s.sum()
'''
Since there are no missing values under part and bundle_id and since s.sum == len(ques.bundle_id), we can surmise that each bundle is included in just one part. Percentage of bundles in different parts, in ascending order:
    part
    1    0.101587
    2    0.168664
    3    0.053354
    4    0.049155
    5    0.564363
    6    0.031132
    7    0.031746

Parts have differnt number of bundles with ~57% in part 5

Comparing percentage of bundles and percentage of questions across parts, one can conclude that for parts 3, 4, 6 and 7 we have relatively more bundles with multiple questions where are for the other parts (1, 2 and 5) we have relatively more bundles with one questions.

Distribution of questions across bundles and parts in skewed; so is the distribution of bundels across parts.
'''

# Data preprocessing
ques['tags'] = ques['tags'].apply(lambda x: str(x).split())
# there is one record with nan in the tags field. We convert 'nan' to '-1' 
# which later aids in converting all tags to integers.
idx = [i for i in range(len(ques)) if 'nan' in ques.tags[i]]
for i in idx:
    ques.loc[i, 'tags'] = [['-1' if x == 'nan' else x for x in ques.tags[i]]]
ques['tags'] = ques['tags'].apply(lambda x: list(map(int, x)))


#%% Lectures
lecs =pd.read_csv(datapath + 'lectures.csv')

lecs.shape
'''(418, 4)'''

lecs.head()
'''Every lecture has only one tag associated with it'''

lecs.columns
'''
    lecture_id: foreign key for the train/test content_id column, when the content type is lecture (1).
    part: top level category code for the lecture.
    tag: one tag codes for the lecture. The meaning of the tags will not be provided.
    type_of: brief description of the core purpose of the lecture
'''

lecs.nunique(axis = 0)
'''
unique counts in each column:
    lecture_id    418
    tag           151
    part            7
    type_of         4
'''

lecs['type_of'].unique()
'''The lectures are of 4 types: ['concept', 'solving question', 'intention', 'starter']'''

s = lecs.groupby('part')['lecture_id'].count()
s.sum()
s/s.sum()
'''
Since s.sum = len(lecs.lecture_id), we can surmise that each lecture is included in only one part.
Percentage of lectures in different parts:
    part
    1    0.129187
    2    0.133971
    3    0.045455
    4    0.074163
    5    0.342105
    6    0.198565
    7    0.076555
    
Part 5 is 'special'; it has the most number of questions, the most number of bundles and the most number of lectures.
'''

s = lecs.groupby('tag')['part'].count()
s.value_counts(sort = False)
'''
A tag may be associated in multiple parts. The distribution of tags associated with multiple parts:
    # part
    1    28
    2    42
    3    41
    4    26
    5     6
    6     7
    7     1
    
For e.g. there are 28 tags that are associated with only one part where as 7 tags are associated with 6
'''

ques.groupby('part')['question_id'].count().corr(lecs.groupby('part')['lecture_id'].count())
# 0.8342
'''Parts with more number of questions also have more number of lectures'''


#%% Tags
# get set of unique tags in lectures
lecs_tags = set(lecs.tag)
# 151 unique tags in lectures

# get set of unique tags in questions
ques_tags = []
ques.tags.apply(lambda x: ques_tags.extend(x))
ques_tags = set(ques_tags)
# 189 unique tags in questions

tags_in_q_not_in_l = ques_tags - lecs_tags
# 38 tags (including the -1 set earlier) are in questions but not in lectures

q_with_tags_not_in_l = []
for i in range(len(ques)):
    for x in ques.loc[i, 'tags']:
        if x in tags_in_q_not_in_l and i not in q_with_tags_not_in_l:
            q_with_tags_not_in_l.append(i)
            
q_subset = ques.iloc[q_with_tags_not_in_l, :]

s = q_subset.groupby('part')['question_id'].count()
s.sum()
s.value_counts(normalize = True, sort = False)

listening_tags = []
for p in np.arange(1, 5, 1):
    ques.query('part == @p').tags.apply(lambda x: listening_tags.extend(x))
    
listening_tags = set(listening_tags)

reading_tags = []
for p in np.arange(5, 8, 1):
    ques.query('part == @p').tags.apply(lambda x: reading_tags.extend(x))
    
reading_tags = set(reading_tags)

reading_tags.intersection(listening_tags)
# there is just one tag [162] common across the listening and the reading parts. In other words a particular question is either "measuring" a listening ability or a reading ability and never both

# Common tags across listening parts
tags_by_part = []
for p in np.arange(1, 5, 1):
    _tags = []
    ques.query('part == @p').tags.apply(lambda x: _tags.extend(x))
    _tags = set(_tags)
    tags_by_part.append(_tags)
    
for i in np.arange(4):
    for j in np.arange(i+1, 4, 1):
        print('Intersection parts %i and %i: ' % (i+1, j+1))
        print(tags_by_part[i].intersection(tags_by_part[j]))
        
'''
Intersection parts 1 and 2: 
{162, 38, 102, 81, 92, 29}
Intersection parts 1 and 3: 
{162, 38, 102, 81, 92, 29}
Intersection parts 1 and 4: 
{162, 38, 102, 81, 92, 29}
Intersection parts 2 and 3: 
{162, 38, 102, 81, 92, 29}
Intersection parts 2 and 4: 
{162, 38, 102, 81, 92, 29}
Intersection parts 3 and 4: 
{161, 162, 38, 102, 136, 74, 106, 81, 82, 113, 29, 92, 157}
'''

# Common tags across reading parts
tags_by_part = []
for p in np.arange(5, 8, 1):
    _tags = []
    ques.query('part == @p').tags.apply(lambda x: _tags.extend(x))
    _tags = set(_tags)
    tags_by_part.append(_tags)
    
for i in np.arange(3):
    for j in np.arange(i+1, 3, 1):
        print('Intersection parts %i and %i: ' % (i+5, j+5))
        print(tags_by_part[i].intersection(tags_by_part[j]))
        
'''
Intersection parts 5 and 6: 
{128, 1, 4, 133, 132, 7, 8, 134, 14, 147, 23, 151, 24, 26, 152, 156, 28, 25, 159, 33, 166, 168, 170, 43, 44, 173, 45, 47, 172, 177, 49, 179, 180, 53, 54, 55, 181, 52, 182, 58, 60, 64, 65, 72, 73, 75, 79, 80, 85, 89, 91, 95, 96, 174, 109, 175, 48, 115, 116, 123, 125, 127}
Intersection parts 5 and 7: 
set()
Intersection parts 6 and 7: 
{162}
'''

#%% Training dataset insights
'''
row_id: (int64) ID code for the row.

timestamp: (int64) the time in milliseconds between this user interaction and the first event completion from that user.

user_id: (int32) ID code for the user.

content_id: (int16) ID code for the user interaction

content_type_id: (int8) 0 if the event was a question being posed to the user, 1 if the event was the user watching a lecture.

task_container_id: (int16) Id code for the batch of questions or lectures. For example, a user might see three questions in a row before seeing the explanations for any of them. Those three would all share a task_container_id.

user_answer: (int8) the user's answer to the question, if any. Read -1 as null, for lectures.

answered_correctly: (int8) if the user responded correctly. Read -1 as null, for lectures.

prior_question_elapsed_time: (float32) The average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures in between. Is null for a user's first question bundle or lecture. Note that the time is the average time a user took to solve each question in the previous bundle.

prior_question_had_explanation: (bool) Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Typically the first several questions a user sees were part of an onboarding diagnostic test where they did not get any feedback.

'''

sum(1 for row in open(datapath + 'train.csv', 'r'))
# There are 101230333 - 1 = 101,230,332 records in the training dataset

reader = pd.read_csv(datapath + 'train.csv', usecols = ['user_id'], 
                     chunksize = 100000, memory_map = True)
users = []
for chunk in reader:
    users_ = chunk.user_id.unique().tolist()
    users = users + users_
    users = list(set(users))
# There are 393656 unique users in the training dataset

# Changing the storage format for train dataset from csv to h5 for IO efficiency gain

# Columns to read from train dataset
usecols = ['row_id', 'user_id', 'content_id', 'content_type_id', 
           'answered_correctly', 'prior_question_had_explanation']

reader = pd.read_csv(datapath + 'train.csv', usecols = usecols,
                     chunksize = 100000, memory_map = True)

for chunk in reader:
    chunk.to_hdf('./data/train.h5', key = 'df', mode = 'a', append = True, format = 'table')

tables.file._open_files.close_all()
