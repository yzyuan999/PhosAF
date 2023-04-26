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
    pho_type = ((file_path_train.split('/')[-1]).split('.')[0]).split('_')[0]  
    
    one_hot_train = np.load(file = r"./feature_data/"+pho_type+"_train_onehot.npy")
    pssm_train = np.load(file = r"./feature_data/"+pho_type+"_train_pssm.npy")
    dist_train = np.load(file = r"./feature_data/"+pho_type+"_train_distancemap.npy")
    angle_train = np.load(file = r"./feature_data/"+pho_type+"_train_angle.npy")
    acc_train = np.load(file = r"./feature_data/"+pho_type+"_train_sasa.npy")
    Y_train = np.load(file = r"./feature_data/"+pho_type+"_train_label.npy")
    one_hot_test = np.load(file = r"./feature_data/"+pho_type+"_test_onehot.npy")
    pssm_test = np.load(file = r"./feature_data/"+pho_type+"_test_pssm.npy")
    dist_test = np.load(file = r"./feature_data/"+pho_type+"_test_distancemap.npy")
    angle_test = np.load(file = r"./feature_data/"+pho_type+"_test_angle.npy")
    acc_test = np.load(file = r"./feature_data/"+pho_type+"_test_sasa.npy")
    Y_test = np.load(file = r"./feature_data/"+pho_type+"_test_label.npy")
    
    nb_epochs=200
    '''
    model_fusion = model_PhosAF(one_hot_train,pssm_train,dist_train,angle_train,acc_train,Y_train,nb_epochs,
                                one_hot_test,pssm_test,dist_test,angle_test,acc_test,Y_test,
                                weights=background_weight)
   '''
    model_fusion = model_PhosAF(one_hot_train,pssm_train,dist_train,angle_train,acc_train,Y_train,nb_epochs,
                                one_hot_test,pssm_test,dist_test,angle_test,acc_test,Y_test,
                                #weights=background_weight
                                )      
    
    #pho_type = ((file_path_train.split('/')[-1]).split('.')[0]).split('_')[0]  
        
    model_fusion.save_weights(predictFrame+'_best_weights_'+pho_type+'.h5',overwrite=True)
    
if __name__ == '__main__':
    file_path_train = r'./data/st_train_data.csv' 
    train_phosaf(file_path_train,predictFrame= 'general',background_weight = None)    


if __name__ == '__main__':
   file_path_train = r'./data/AGC_train_data.csv' 
   
   train_phosaf(file_path_train,predictFrame= 'kinase',background_weight = r'./models/general_best_weights_st.h5')    

