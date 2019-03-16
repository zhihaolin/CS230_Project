#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:31:13 2019

@author: kristen
"""

import pandas as pd

DATA_FILE = 'train.csv'
NEW_FILE = 'test_file_seqgan_2019-03-15-2047.txt'
data = pd.read_csv(DATA_FILE)
data['generate'] = '0'
new_data = pd.read_table(NEW_FILE,names=['comment_text'], header=None)
new_data['id'] = '0' 
new_data['toxic'] = '0'
new_data['severe_toxic'] = '0'
new_data['obscene'] = '0'
new_data['threat'] = '0'
new_data['insult'] = '0'
new_data['identity_hate'] = '0'
new_data['generate'] = '1'
cols = ['id']  + ['comment_text']+['toxic']+['severe_toxic']+['obscene']+['threat']+ ['insult']+['identity_hate']+ ['generate']
newdata = new_data[cols]
frames = [data, newdata]
newtrain = pd.concat(frames)
newtrain.to_csv('/Users/kristen/Desktop/newtrain.csv')





