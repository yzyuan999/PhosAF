# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 21:22:38 2023

@author: DIY
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_encoding import get_train_contactmap,get_train_angle,get_train_sasa,get_train_pssm,get_ind_contactmap,get_ind_angle,get_ind_sasa,get_ind_pssm
from functools import reduce


def get_train_sample(file_path_train,file_path_3ddata,file_path_angle,file_path_sasa,file_path_PSSM,lenth):
    trainseq = pd.read_csv(file_path_train)
    train_contactmap=get_train_contactmap(file_path_train,file_path_3ddata,lenth)
    train_angle=get_train_angle(file_path_train,file_path_angle,lenth)
    train_sasa=get_train_sasa(file_path_train,file_path_sasa,lenth)
    train_pssm=get_train_pssm(file_path_train,file_path_PSSM)
    train = [trainseq, train_contactmap, train_angle, train_sasa, train_pssm]
    df_train = reduce(lambda x, y: pd.merge(x, y, on="name", how="outer"),train)
    df_train=df_train.dropna()
    order=['name', 'seq','pssm','angle', 'contactmap', 'sasa', 'label']
    df_train=df_train[order]
    return df_train

def get_ind_sample(file_path_ind,file_path_3ddata,file_path_angle,file_path_sasa,file_path_PSSM,lenth):
    indseq = pd.read_csv(file_path_ind)
    ind_contactmap=get_ind_contactmap(file_path_ind,file_path_3ddata,lenth)
    ind_angle=get_ind_angle(file_path_ind,file_path_angle,lenth)
    ind_sasa=get_ind_sasa(file_path_ind,file_path_sasa,lenth)
    ind_pssm=get_ind_pssm(file_path_ind,file_path_PSSM)
    ind = [indseq, ind_contactmap, ind_angle, ind_sasa, ind_pssm]
    df_ind= reduce(lambda x, y: pd.merge(x, y, on="name", how="outer"),ind)
    df_ind=df_ind.dropna()
    order=['name', 'seq','pssm','angle', 'contactmap', 'sasa', 'label']
    df_ind=df_ind[order]
    return df_ind

def embedding_train_test(file_path_train):      
    train_set=get_train_sample(file_path_train,r'./feature/3ddata',
                 r'./feature/dssp_sincos_angle',r"./feature/dssp_sasa",
                 r"./feature/pssm",20)
    np.random.seed(1)
    final_data=train_set.sample(frac=1)
    Y=final_data['label']
    X1=final_data['angle']
    X_angle=np.zeros([len(X1),41,4])
    for p,q in enumerate(X1):
        #print(p,q)        
        for m,n in enumerate(q):
            #print(m,n)
            for s,t in enumerate(n):
                #print(s,t)         
                X_angle[p,m,s]=float(t)   
    
    X2=final_data['seq']  
    labels=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    seq_onehot=np.zeros([len(X2),41,20])
    seq_to_idx = dict((seq, number) for number, seq in enumerate(labels))
    for i,acid in enumerate(X2):
        for j,m in enumerate(acid):
            seq_onehot[i,j,seq_to_idx[m]]=1
    X_onehot=seq_onehot
    
    X3=final_data['contactmap']
    X_dist=np.zeros([len(X3),41,41])
    for a,b in enumerate(X3):
        #print(a,b)
        for c,d in enumerate(b):
            #print(c,d)
            for e,f in enumerate(d):
                #print(e,f)    
                X_dist[a,c,e]=float(f)
    
    X5=final_data['sasa']
    X_acc=np.zeros([len(X5),41,1])
    for k,l in enumerate(X5):
        #print(k,l)
        for y,z in enumerate(l):
            X_acc[k,y,0]=float(z)
    
    X6=final_data['pssm']  
    X_pssm=np.zeros([len(X6),41,20])
    for g,h in enumerate(X6):   
        #print(g,h)
        for o,r in enumerate(h):
           #print(o,r)
           for u,v in enumerate(r):
               #print(u,v)
               X_pssm[g,o,u]=float(v)        
    
    X=np.concatenate((X_onehot,                      
                      X_angle,
                      X_dist,
                      X_acc,
                      X_pssm),
                     axis=2) 

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    
    one_hot_train=np.zeros([len(X_train),41,20])
    angle_train=np.zeros([len(X_train),41,4])
    dist_train=np.zeros([len(X_train),41,41])
    acc_train=np.zeros([len(X_train),41,1])
    pssm_train=np.zeros([len(X_train),41,20])
    for m,n in enumerate(X_train):
        for s,t in enumerate(n):
            for u,v in enumerate(t):
                if u<20:
                    one_hot_train[m,s,u]=float(v)
                elif u>19 and u<24:
                    angle_train[m,s,u-20]=float(v)
                elif u>23 and u<65:
                    dist_train[m,s,u-24]=float(v)
                elif u>64 and u<66:
                    acc_train[m,s,u-65]=float(v)
                else:
                    pssm_train[m,s,u-66]=float(v)
                            
    one_hot_test=np.zeros([len(X_test),41,20])
    angle_test=np.zeros([len(X_test),41,4])
    dist_test=np.zeros([len(X_test),41,41])
    acc_test=np.zeros([len(X_test),41,1])
    pssm_test=np.zeros([len(X_test),41,20])
    
    for m,n in enumerate(X_test):
        for s,t in enumerate(n):
            for u,v in enumerate(t):
                if u<20:
                    one_hot_test[m,s,u]=float(v)            
                elif u>19 and u<24:
                    angle_test[m,s,u-20]=float(v)
                elif u>23 and u<65:
                    dist_test[m,s,u-24]=float(v) 
                elif u>64 and u<66:
                     acc_test[m,s,u-65]=float(v)
                else:
                    pssm_test[m,s,u-66]=float(v) 
    return one_hot_train,pssm_train,dist_train,angle_train,acc_train,Y_train,one_hot_test,pssm_test,dist_test,angle_test,acc_test,Y_test
    
def embedding_independent(file_path_ind):   
    independent_set=get_ind_sample(file_path_ind,
                 r'./feature/3ddata',r'./feature/dssp_sincos_angle',r"./feature/dssp_sasa",
                 r"./feature/pssm",20)
    np.random.seed(1)
    final_data=independent_set.sample(frac=1)
    Y_independent=final_data['label']
    X1=final_data['angle']
    X_independent_angle=np.zeros([len(X1),41,4])
    for p,q in enumerate(X1):
        #print(p,q)        
        for m,n in enumerate(q):
            #print(m,n)
            for s,t in enumerate(n):
                #print(s,t)         
                X_independent_angle[p,m,s]=float(t)   
    
    X2=final_data['seq']  
    labels=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    seq_onehot=np.zeros([len(X2),41,20])
    seq_to_idx = dict((seq, number) for number, seq in enumerate(labels))
    for i,acid in enumerate(X2):
        for j,m in enumerate(acid):
            seq_onehot[i,j,seq_to_idx[m]]=1
    X_independent_onehot=seq_onehot
    
    X3=final_data['contactmap']
    X_independent_dist=np.zeros([len(X3),41,41])
    for a,b in enumerate(X3):
        #print(a,b)
        for c,d in enumerate(b):
            #print(c,d)
            for e,f in enumerate(d):
                #print(e,f)    
                X_independent_dist[a,c,e]=float(f)
    
    X5=final_data['sasa']
    X_independent_acc=np.zeros([len(X5),41,1])
    for k,l in enumerate(X5):
        #print(k,l)
        for y,z in enumerate(l):
            X_independent_acc[k,y,0]=float(z)
    
    X6=final_data['pssm']  
    X_independent_pssm=np.zeros([len(X6),41,20])
    for g,h in enumerate(X6):   
        #print(g,h)
        for o,r in enumerate(h):
           #print(o,r)
           for u,v in enumerate(r):
               #print(u,v)
               X_independent_pssm[g,o,u]=float(v)        
    
    return X_independent_onehot,X_independent_pssm,X_independent_dist,X_independent_angle,X_independent_acc,Y_independent
