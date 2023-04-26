# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:53:38 2023

@author: DIY
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def model_PhosAF(one_hot_train,pssm_train,dist_train,angle_train,acc_train,Y_train,nb_epochs,
                 #one_hot_test,pssm_test,dist_test,angle_test,acc_test,Y_test,
                 weights=None):
    #nb_epochs = 200
    #nb_batch_size = 512
    nb_batch_size = 32
    init_form = 'he_normal'
    weight_decay = 0
    nb_filter1 = 128
    nb_filter2 = 256
    kernel_size1 = 1
    kernel_size2 = 3
    kernel_size3 = 5
    multi_number = 128
    multi_head = 8
    dropout_rate1 = 0.3
    nb_dense1 = 128
    nb_dense2 = 256
    nb_dense3 = 64
    nb_dense4 = 32
    nb_class = 2
    lr_1 = 0.2
    ###########model###########
    from phosaf import PhosAF
    model_fusion = PhosAF(one_hot_train,nb_filter1,nb_filter2,kernel_size1,kernel_size2,kernel_size3,
                          init_form,weight_decay,dropout_rate1,multi_number,multi_head,lr_1,
                          pssm_train,nb_dense1,nb_dense2,nb_dense3,
                          dist_train,angle_train,acc_train,nb_dense4,nb_class)
    model_fusion.summary()
    model_fusion.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])    
    #earlystop_callback=EarlyStopping(monitor='val_accuracy',min_delta=0.00001,mode='max',verbose=2,patience=50)
    
    if weights is not None:
        model_fusion.load_weights(weights)
        model2 = model_fusion
        model2.load_weights(weights)
        for num in range(len(model2.layers) - 1):
            model_fusion.layers[num].set_weights(model2.layers[num].get_weights())        
        model_fusion.fit([one_hot_train,
                         pssm_train,
                         dist_train,
                         angle_train,
                         acc_train
                         ],to_categorical(Y_train),
                        epochs=nb_epochs,batch_size=nb_batch_size,
                        #validation_data=([one_hot_test,
                                          #pssm_test,
                                          #dist_test,
                                          #angle_test,
                                         # acc_test
                                          #],to_categorical(Y_test)), 
                        #validation_freq=1
                        #callbacks=[earlystop_callback]  
                        )
    else:
        model_fusion.fit([one_hot_train,
                         pssm_train,
                         dist_train,
                         angle_train,
                         acc_train
                         ],to_categorical(Y_train),
                        epochs=nb_epochs,batch_size=nb_batch_size,
                        #validation_data=([one_hot_test,
                                          #pssm_test,
                                          #dist_test,
                                          #angle_test,
                                          #acc_test
                                          #],to_categorical(Y_test)), 
                        #validation_freq=1
                        #callbacks=[earlystop_callback]
                        )
    
    return model_fusion
    
 