# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:06:00 2023

@author: DIY
"""

import numpy as np
import pandas as pd
from embedding_train_ind import embedding_train_test,embedding_independent
from model_phosaf import model_PhosAF

def train_phosaf(file_path_train,predictFrame,background_weight=None):
    '''
    predictFrame : 'general' or 'kinase'
    background_weight : a '.h5' file, used to train specific-kinase protein
       
    '''
    one_hot_train = embedding_train_test(file_path_train)[0]
    pssm_train = embedding_train_test(file_path_train)[1]
    dist_train = embedding_train_test(file_path_train)[2]
    angle_train = embedding_train_test(file_path_train)[3]
    acc_train = embedding_train_test(file_path_train)[4]
    Y_train = embedding_train_test(file_path_train)[5]
    one_hot_test = embedding_train_test(file_path_train)[6]
    pssm_test = embedding_train_test(file_path_train)[7]
    dist_test = embedding_train_test(file_path_train)[8]
    angle_test = embedding_train_test(file_path_train)[9]
    acc_test = embedding_train_test(file_path_train)[10]
    Y_test = embedding_train_test(file_path_train)[11]
    nb_epochs=200
    
    model_fusion = model_PhosAF(one_hot_train,pssm_train,dist_train,angle_train,acc_train,Y_train,nb_epochs,
                                #one_hot_test,pssm_test,dist_test,angle_test,acc_test,Y_test,
                                weights=background_weight)
      
    
    pho_type = ((file_path_train.split('/')[-1]).split('.')[0]).split('_')[0]  
        
    model_fusion.save_weights(predictFrame+'_best_weights_'+pho_type+'.h5',overwrite=True)
    
if __name__ == '__main__':
    file_path_train = r'./data/y_train_data.csv' 
    train_phosaf(file_path_train,predictFrame= 'general',background_weight = None)    


if __name__ == '__main__':
   file_path_train = r'./data/AGC_train_data.csv' 
   
   train_phosaf(file_path_train,predictFrame= 'kinase',background_weight = r'./models/general_best_weights_st.h5')    

