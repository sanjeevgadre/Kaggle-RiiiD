#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 08:59:37 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

#%% Helper Variables
DATAPATH = './data/'

#%% Helper Functions
# OneHotEncoder to encode the questions part number
part_enc = OneHotEncoder(categories = [np.arange(1, 8, 1)], dtype = 'int', sparse = False)

#%% Get Scoring data
userscores = np.genfromtxt(DATAPATH + 'userscores.csv', delimiter = ',')
ques = np.genfromtxt(DATAPATH + 'ques.csv', delimiter = ',')
model = xgb.Booster(model_file = DATAPATH + 'xgbmodel.bin')

# mean userscores to use when a new user is encountered
mean_userscores = np.mean(userscores[:, 1:8], axis = 0).reshape(1, -1)

train = pd.read_csv(DATAPATH + 'train_proc_train.csv', nrows = 5)

#%% Process Test data
test = pd.read_csv(DATAPATH + 'example_test.csv')
sample_submission = pd.read_csv(DATAPATH + 'example_sample_submission.csv')
# Drop unwanted rows
test = test.drop(index = test.loc[test['content_type_id'] == 1, :].index)
# Drop unwanted columns
test = test.drop(columns = ['timestamp', 'content_type_id', 'task_container_id',
                            'prior_question_elapsed_time',                        
                            'prior_group_answers_correct', 'prior_group_responses'])
# test_cols : ['row_id', 'group_num', 'user_id', 'content_id', 'prior_question_had_explanation']
# Eliminating nans
test.loc[test['prior_question_had_explanation'].isna(), 'prior_question_had_explanation'] = False
test['prior_question_had_explanation'] = test['prior_question_had_explanation'].astype('int')
# Convert to numpy array
test = test.to_numpy()
'''
Add new columns required --> ['score_1', score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 
                              'part', 'correct_attempt_prob'] i.e. 9 new columns

test_cols : ['row_id', 'group_num', 'user_id', 'content_id', 'prior_question_had_explanation',
             'score_1', score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 'part',
             'correct_attempt_prob']
'''
test = np.concatenate((test, np.zeros((test.shape[0], 9))), axis = 1)
# To the test dataset add the appropriate part, correct_attempt_prob and userscores
for i in range(len(test)):
    # part and correct_attempt_prob
    test[i, 12:] = ques[np.where(ques[:, 0] == test[i, 3])[0], [1, 3]]
    # userscores
    # is the user a "new" user
    if np.isin(test[i, 2], userscores[:, 0], assume_unique = True):
        test[i, 5:12] = userscores[np.where(userscores[:, 0] == test[i,2])[0], 1:8]
    else:
        test[i, 5:12] = mean_userscores

# OneHotEncode the part number
encoded_part = part_enc.fit_transform(test[:, 12].reshape(-1, 1))
# Drop the user_id, content_id and part_number columns and add the encoded part columns
test = np.delete(test, [2, 3, 12], 1)
test = np.concatenate((test, encoded_part), axis = 1)
'''
test_cols : ['row_id', 'group_num', 'prior_question_had_explanation',
             'score_1', score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7',
             correct_attempt_prob', 'part_1', 'part_2', 'part_3', 'part_4', 'part_5',
             'part_6', 'part_7']
'''
# Make predictions
dtest = xgb.DMatrix(test[:, 2:])
probs = model.predict(dtest)[:, 1]
predict_df = pd.DataFrame(
                    data = {'row_id' : test[:, 0],
                            'answered_correctly' : probs,
                            'group_num' : test[:, 1]
                            }
                    )





