# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:14:58 2023

@author: DIY
"""
import numpy as np
import pandas as pd
import os
import csv
##############read negative seq##############
def read_negative_seq(file_path_neg):
    neg = open(file_path_neg, 'r')
    neg = neg.readlines()
    neg_name = []
    neg_seq = []
    for i in range(len(neg)):
        if i % 2 == 0:
            neg_name.append(neg[i][:-1])
        else:
            neg_seq.append(neg[i][:-1])
    neg_name = (np.asarray(neg_name)).reshape(-1, 1)
    neg_seq = (np.asarray(neg_seq)).reshape(-1, 1)
    number = len(neg_name)
    neg_label = (np.zeros((number), dtype=int)).reshape(-1, 1)###### 
    neg_array = np.transpose(np.vstack((neg_name.T, neg_seq.T, neg_label.T)))
    neg_df = pd.DataFrame(neg_array)
    neg_df.columns = ['name', 'seq', 'label']
    return neg_df

def read_protein_ss(file_path_neg,file_path_ss):
    '''
    file_path_ss : A folder contains the predicted secondary structure information of each protein, 
    where each protein corresponds to a CSV file, the first column is the residue name,
    and the second column is the secondary structure information.
    '''
    data_ss_dict = {}
    neg_name = read_negative_seq(file_path_neg)['name'].tolist()
    for root, dirs, files in os.walk(file_path_ss):
        for p in range(len(files)):
            q = file_path_ss+'\\'+files[p]
            fileSS_name = os.path.basename(q)
            fileSS_name = fileSS_name.split('_')[0]
            data_SS = pd.read_csv(q, header=0, encoding="gbk")
            data_SS = np.array(data_SS)
            for key in neg_name:
                if fileSS_name in key[4:10]:
                    position = key[13:17]
                    position = int(position)
                    data_s_dict = {}
                    if data_SS[position-1] == 'H' or data_SS[position-1] == 'G' or data_SS[position-1] == 'I':
                        data_s_dict[key] = 'helix'
                    data_ss_dict.update(data_s_dict)
    return data_ss_dict

def screen_negative_samples(file_path_neg,file_path_ss):
    '''
    file_path_ss : A folder contains the predicted secondary structure information of each protein, 
    where each protein corresponds to a CSV file, the first column is the residue name,
    and the second column is the secondary structure information.
    '''
    data_ss_dict = {}
    neg_name = read_negative_seq(file_path_neg)['name'].tolist()
    for root, dirs, files in os.walk(file_path_ss):
        for p in range(len(files)):
            q = file_path_ss+'\\'+files[p]
            fileSS_name = os.path.basename(q)
            fileSS_name = fileSS_name.split('_')[0]
            data_SS = pd.read_csv(q, header=0, encoding="gbk")
            data_SS = np.array(data_SS)
            for key in neg_name:
                if fileSS_name in key[4:10]:
                    position = key[13:17]
                    position = int(position)
                    data_s_dict = {}
                    if data_SS[position-1] == 'H' or data_SS[position-1] == 'G' or data_SS[position-1] == 'I':
                        data_s_dict[key] = 'helix'
                    data_ss_dict.update(data_s_dict)
    with_ss_neg = pd.DataFrame([data_ss_dict]).T
    with_ss_neg = with_ss_neg.reset_index()
    with_ss_neg.columns = ['name', 'ss']
    helix_name = pd.DataFrame(with_ss_neg[with_ss_neg['ss'] == 'helix']['name'])
    neg_df=read_negative_seq(file_path_neg)
    final_neg = pd.merge(helix_name, neg_df, on='name')   
    return final_neg
