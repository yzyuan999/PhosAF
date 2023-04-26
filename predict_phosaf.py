# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:11:57 2023

@author: DIY
"""

import numpy as np
import pandas as pd
from embedding_train_ind import embedding_train_test,embedding_independent
from model_phosaf import model_PhosAF



def predict_phosaf(file_path_ind,predictFrame,kinase=None):
    '''
    predictFrame : 'general' or 'kinase'
       
    ''' 
    pho_type = ((file_path_ind.split('/')[-1]).split('.')[0]).split('_')[0]   
    X_independent_onehot = np.load(file = r"./feature_data/"+pho_type+"_independent_onehot.npy")
    X_independent_pssm = np.load(file = r"./feature_data/"+pho_type+"_independent_pssm.npy")
    X_independent_dist = np.load(file = r"./feature_data/"+pho_type+"_independent_distancemap.npy")
    X_independent_angle = np.load(file = r"./feature_data/"+pho_type+"_independent_angle.npy")
    X_independent_acc = np.load(file = r"./feature_data/"+pho_type+"_independent_sasa.npy")
    Y_independent = np.load(file = r"./feature_data/"+pho_type+"_independent_label.npy")
   
    nb_epochs = 0
    model_fusion = model_PhosAF(X_independent_onehot,X_independent_pssm,X_independent_dist,X_independent_angle,
                                X_independent_acc,
                                Y_independent,
                                nb_epochs)
    
    #predictFrame = 'general'
    if predictFrame == 'general':
        outputfile = 'general_{:s}'.format(pho_type)
        if pho_type == 'st':
            model_weight = r'./models/general_best_weights_st.h5'
        if pho_type == 'y':
            model_weight = r'./models/general_best_weights_y.h5'
    if predictFrame == 'kinase':
        outputfile = 'kinase_{:s}'.format(kinase)
        model_weight = './models/kinase_best_weights_{:s}.h5'.format(kinase)
        
        model_fusion.load_weights(model_weight)
    
    predict_result = model_fusion.predict([X_independent_onehot,X_independent_pssm,X_independent_dist,
                                           X_independent_angle,X_independent_acc])
    
    predict_class = np.argmax(predict_result,axis=1)
    #predict_score = predict_result[:,1]

    result = pd.DataFrame(predict_class,columns=['pred_label'])
    result.to_csv(outputfile + "_predictresults.csv", index=False,sep='\t')


if __name__ == '__main__':
    file_path_ind = r'./data/CDK_independent_data.csv' 
    predict_phosaf(file_path_ind,predictFrame = 'kinase',kinase='CDK')
        
    
if __name__ == '__main__':
    file_path_ind = r'./data/st_independent_data.csv'  
    predict_phosaf(file_path_ind,predictFrame = 'general',kinase=None)    
    