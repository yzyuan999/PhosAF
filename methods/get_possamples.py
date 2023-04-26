# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:54:26 2023

@author: DIY
"""

import numpy as np
import pandas as pd
import os
import csv
##############read positive seq##############
def read_positive_seq(file_path_pos):
    '''
    Parameters
    ----------
    file_path_pos : A txt file containing protein name, residue, position and sequence fragment information.
    take a protein for example:
        >sp|P51451|Y|69
        PPPPDEHLDEDKHFVVALYDYTAMNDRDLQMLKGEKLQVLK
    '''
    pos = open(file_path_pos, 'r')
    pos = pos.readlines()
    pos_name = []
    pos_seq = []
    for i in range(len(pos)):
        if i % 2 == 0:
            pos_name.append(pos[i][:-1])
        else:
            pos_seq.append(pos[i][:-1])
    pos_name = (np.asarray(pos_name)).reshape(-1, 1)
    pos_seq = (np.asarray(pos_seq)).reshape(-1, 1)
    number = len(pos_name)
    pos_label = (np.ones((number), dtype=int)).reshape(-1, 1)###### 
    pos_array = np.transpose(np.vstack((pos_name.T, pos_seq.T, pos_label.T)))
    pos_df = pd.DataFrame(pos_array)
    pos_df.columns = ['name', 'seq', 'label']
    return pos_df



