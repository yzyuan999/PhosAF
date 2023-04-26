# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:31:06 2023

@author: DIY
"""

import pandas as pd
import numpy as np
import os
import copy
import csv
##############train feature eoncding##############
##############contact map##############
def get_train_contactmap(file_path_train,file_path_3ddata,lenth):
    '''
    file_path_3ddata :A folder contains the predicted 3d structure information of each protein, 
    where each protein corresponds to a CSV file,named 'proteinname_3ddata', 
    the first three column are three-dimensional coordinates of residues,
    and the last column is the value of plddt. 
    '''    
    data_3d_dict = {}
    trainname = pd.read_csv(file_path_train)['name']
    for root, dirs, files in os.walk(file_path_3ddata):
        for p in range(len(files)):
            q = file_path_3ddata+'\\'+files[p]
            file3d_name = os.path.basename(q)
            file3d_name = file3d_name.split('_')[0]
            data_3d = pd.read_csv(q, header=0, encoding="gbk")
            data_3d = np.array(data_3d)
            for key in trainname:
                if file3d_name in key[4:10]:
                    position = key[13:17]
                    position = int(position)
                    Max = len(data_3d)
                    data3d_dict = {}
                    if position-(lenth+1) < 0:
                        data3d_dict[key] = data_3d[0:position+lenth]
                    elif position+lenth > Max:
                        data3d_dict[key] = data_3d[position-(lenth+1):Max]
                    elif position-(lenth+1) < 0 and position+lenth > Max:
                        data3d_dict[key] = data_3d[0:Max]
                    else:
                        data3d_dict[key]=data_3d[position-(lenth+1):position+lenth]
                    data_3d_dict.update(data3d_dict)
    
    data_3d_change=copy.deepcopy(data_3d_dict)
    for a in list(data_3d_change.keys()):
        data_3d_array=data_3d_change[a]
        if data_3d_array.dtype != "float64":
            data_3d_change.pop(a)
        else:
            data_3d_change                
    
    #delete filled sequence data  
    seq_lenth=41
    data_3d_new=copy.deepcopy(data_3d_change)            
    for name in list(data_3d_new.keys()):
        #print(name)
        array_3d=data_3d_new[name]
        number_aa=array_3d.shape[0]
        if number_aa != seq_lenth:
            data_3d_new.pop(name)
        else:
            data_3d_new
    
    #obtain 3D coordinate
    seq_3d_dict=copy.deepcopy(data_3d_new) 
    for name_1 in list(seq_3d_dict.keys()):
        seq_3d=seq_3d_dict[name_1]
        seq_3d_new=np.delete(seq_3d,-1,axis=1)
        seq_3d_dict[name_1]=seq_3d_new
        
    ####calculate distance and contact map
    duichen_3d_dict=copy.deepcopy(seq_3d_dict)    
    for name_4 in list(duichen_3d_dict.keys()):
        duichen_3d=duichen_3d_dict[name_4]
        g=duichen_3d.shape[0]
        np.set_printoptions(linewidth=99999)
        D = np.zeros([g,g])
        for i in range(g):
            for j in range(i+1, g):
                D[i,j] = np.linalg.norm(duichen_3d[i, :] - duichen_3d[j, :])
                ###if distance >8,irrelevant
                if D[i,j]>8:
                    D[i,j]=0
                else:
                    D[i,j]=1
                D[j,i] = D[i,j]        
            #print(i)               
        duichen_3d_dict[name_4]=D
    duichen_3d_dict = pd.DataFrame([duichen_3d_dict]).T
    duichen_3d_dict = duichen_3d_dict.reset_index()
    duichen_3d_dict.columns = ['name', 'contactmap']
    return duichen_3d_dict

##############angle##############
def get_train_angle(file_path_train,file_path_angle,lenth):
    '''
    file_path_angle :A folder contains the predicted angle information of each protein, 
    where each protein corresponds to a CSV file,named 'proteinname_angle', 
    the six columns are PHI,PSI,sin_PHI,cos_PHI,sin_PSI and	cos_PSI
    '''    
    data_phipsi_dict={}
    trainname = pd.read_csv(file_path_train)['name']
    for root, dirs, files in os.walk(file_path_angle):
        for p in range(len(files)):
            q=file_path_angle+'\\'+files[p]
            #print(q)
           
            file_angle_name = os.path.basename(q)
            file_angle_name = file_angle_name.split('_')[0]
            #obtain angle information
            data_angle=pd.read_csv(q,header=0,encoding="gbk")
            data_angle=data_angle.loc[:,['sin_PHI','cos_PHI','sin_PSI','cos_PSI']]
            data_angle=np.array(data_angle)
            for key in trainname:
                #print(key)
                if file_angle_name in key[4:10]:
                    #print('ok')
                    position=key[13:]
                    position=int(position)
                    Max=len(data_angle)
                    data_angle_dict={}
                    if position-(lenth+1)<0 and position+lenth<Max:
                        data_angle_dict[key]=data_angle[0:(position+lenth)]
                    elif position+lenth>Max and position-(lenth+1)>0:
                        data_angle_dict[key]=data_angle[position-(lenth+1):Max]
                    elif position-(lenth+1)<0 and position+lenth>Max:
                        data_angle_dict[key]=data_angle[0:Max]
                    else:
                        data_angle_dict[key]=data_angle[position-(lenth+1):position+lenth]
                    data_phipsi_dict.update(data_angle_dict)
    
    seq_lenth=41
    data_angle_new=copy.deepcopy(data_phipsi_dict)            
    for name in list(data_angle_new.keys()):
        #print(name)
        array_angle=data_angle_new[name]
        number_aa=array_angle.shape[0]
        if number_aa != seq_lenth:
            data_angle_new.pop(name)
        else:
            data_angle_new  
    data_angle_new = pd.DataFrame([data_angle_new]).T
    data_angle_new = data_angle_new.reset_index()
    data_angle_new.columns = ['name', 'angle']
    return data_angle_new

##############SASA##############
def get_train_sasa(file_path_train,file_path_sasa,lenth):
    '''
    file_path_sasa :A folder contains the predicted sasa information of each protein, 
    where each protein corresponds to a CSV file,named 'proteinname_dsspdata', 
    the last column is sasa
    '''  
    data_SASA_dict={}
    trainname = pd.read_csv(file_path_train)['name']
    for root, dirs, files in os.walk(file_path_sasa):
        for p in range(len(files)):
            q=file_path_sasa+'\\'+files[p]
            #print(q)
            file_sasa_name = os.path.basename(q)
            file_sasa_name = file_sasa_name.split('_')[0]
            data_sasa=pd.read_csv(q,header=0,encoding="gbk")            
            data_sasa=data_sasa.loc[:,['ACC']]
            data_sasa=np.array(data_sasa)
            for key in trainname:
                if file_sasa_name in key[4:10]:
                    position=key[13:]
                    position=int(position)
                    Max=len(data_sasa)
                    data_sasa_dict={}
                    if position-(lenth+1)<0 and position+lenth<Max:
                        data_sasa_dict[key]=data_sasa[0:(position+lenth)]
                    elif position+lenth>Max and position-(lenth+1)>0:
                        data_sasa_dict[key]=data_sasa[position-(lenth+1):Max]
                    elif position-(lenth+1)<0 and position+lenth>Max:
                        data_sasa_dict[key]=data_sasa[0:Max]
                    else:
                        data_sasa_dict[key]=data_sasa[position-(lenth+1):position+lenth]
                    data_SASA_dict.update(data_sasa_dict)
    
    seq_lenth=41
    data_sasa_new=copy.deepcopy(data_SASA_dict)            
    for name in list(data_sasa_new.keys()):       
        array_sasa=data_sasa_new[name]
        number_a=array_sasa.shape[0]
        if number_a != seq_lenth:
            data_sasa_new.pop(name)
        else:
            data_sasa_new  
    data_sasa_new = pd.DataFrame([data_sasa_new]).T
    data_sasa_new = data_sasa_new.reset_index()
    data_sasa_new.columns = ['name', 'sasa']
    return data_sasa_new

##############pssm##############
def get_train_pssm(file_path_train,file_path_pssm):
    '''
    file_path_sasa :A folder contains the pssm information of each protein, 
    where each protein corresponds to a txt file,named 'proteinname_residue_position'
    '''     
    data_PSSM_dict={}
    trainname = pd.read_csv(file_path_train)['name']
    for root, dirs, files in os.walk(file_path_pssm):
        for p in range(len(files)):
            q=file_path_pssm+'\\'+files[p]
            file_pssm_name = os.path.basename(q)
            file_pssm_name = file_pssm_name.split('.')[0]
            file_pssm_name = file_pssm_name.replace("_", "|")        
            pssm_array=np.zeros((41,20),dtype=float)
            f=open(q)
            data_pssm=f.readlines()
            del(data_pssm[41:])
            row=0
            for line in data_pssm:
                pssm_1=line.strip('\n').split(' ')
                pssm_1 = [i for i in pssm_1 if i]
                pssm_array[row:]=pssm_1[2:22]
                row+=1
                for key in trainname:
                    data_pssm_dict={} 
                    if file_pssm_name in key:                   
                        data_pssm_dict[key]=pssm_array                
                        data_PSSM_dict.update(data_pssm_dict)
    data_PSSM_dict = pd.DataFrame([data_PSSM_dict]).T
    data_PSSM_dict = data_PSSM_dict.reset_index()
    data_PSSM_dict.columns = ['name', 'pssm']
    return data_PSSM_dict

##############independent feature eoncding##############
##############contact map##############
def get_ind_contactmap(file_path_ind,file_path_3ddata,lenth):
    '''
    file_path_3ddata :A folder contains the predicted 3d structure information of each protein, 
    where each protein corresponds to a CSV file,named 'proteinname_3ddata', 
    the first three column are three-dimensional coordinates of residues,
    and the last column is the value of plddt. 
    '''    
    data_3d_dict = {}
    indname = pd.read_csv(file_path_ind)['name']
    for root, dirs, files in os.walk(file_path_3ddata):
        for p in range(len(files)):
            q = file_path_3ddata+'\\'+files[p]
            file3d_name = os.path.basename(q)
            file3d_name = file3d_name.split('_')[0]
            data_3d = pd.read_csv(q, header=0, encoding="gbk")
            data_3d = np.array(data_3d)
            for key in indname:
                if file3d_name in key[4:10]:
                    position = key[13:17]
                    position = int(position)
                    Max = len(data_3d)
                    data3d_dict = {}
                    if position-(lenth+1) < 0:
                        data3d_dict[key] = data_3d[0:position+lenth]
                    elif position+lenth > Max:
                        data3d_dict[key] = data_3d[position-(lenth+1):Max]
                    elif position-(lenth+1) < 0 and position+lenth > Max:
                        data3d_dict[key] = data_3d[0:Max]
                    else:
                        data3d_dict[key]=data_3d[position-(lenth+1):position+lenth]
                    data_3d_dict.update(data3d_dict)
    
    data_3d_change=copy.deepcopy(data_3d_dict)
    for a in list(data_3d_change.keys()):
        data_3d_array=data_3d_change[a]
        if data_3d_array.dtype != "float64":
            data_3d_change.pop(a)
        else:
            data_3d_change                
    
    seq_lenth=41
    data_3d_new=copy.deepcopy(data_3d_change)            
    for name in list(data_3d_new.keys()):
        #print(name)
        array_3d=data_3d_new[name]
        number_aa=array_3d.shape[0]
        if number_aa != seq_lenth:
            data_3d_new.pop(name)
        else:
            data_3d_new
    
    seq_3d_dict=copy.deepcopy(data_3d_new) 
    for name_1 in list(seq_3d_dict.keys()):
        seq_3d=seq_3d_dict[name_1]
        seq_3d_new=np.delete(seq_3d,-1,axis=1)
        seq_3d_dict[name_1]=seq_3d_new
        
    duichen_3d_dict=copy.deepcopy(seq_3d_dict)    
    for name_4 in list(duichen_3d_dict.keys()):
        duichen_3d=duichen_3d_dict[name_4]
        g=duichen_3d.shape[0]
        np.set_printoptions(linewidth=99999)
        D = np.zeros([g,g])
        for i in range(g):
            for j in range(i+1, g):
                D[i,j] = np.linalg.norm(duichen_3d[i, :] - duichen_3d[j, :])
                if D[i,j]>8:
                    D[i,j]=0
                else:
                    D[i,j]=1
                D[j,i] = D[i,j]   
                           
        duichen_3d_dict[name_4]=D
    duichen_3d_dict = pd.DataFrame([duichen_3d_dict]).T
    duichen_3d_dict = duichen_3d_dict.reset_index()
    duichen_3d_dict.columns = ['name', 'contactmap']
    return duichen_3d_dict

##############angle##############
def get_ind_angle(file_path_ind,file_path_angle,lenth):
    '''
    file_path_angle :A folder contains the predicted angle information of each protein, 
    where each protein corresponds to a CSV file,named 'proteinname_angle', 
    the six columns are PHI,PSI,sin_PHI,cos_PHI,sin_PSI and	cos_PSI
    '''    
    data_phipsi_dict={}
    indname = pd.read_csv(file_path_ind)['name']
    for root, dirs, files in os.walk(file_path_angle):
        for p in range(len(files)):
            q=file_path_angle+'\\'+files[p]
            file_angle_name = os.path.basename(q)
            file_angle_name = file_angle_name.split('_')[0]
           
            data_angle=pd.read_csv(q,header=0,encoding="gbk")
            data_angle=data_angle.loc[:,['sin_PHI','cos_PHI','sin_PSI','cos_PSI']]
            data_angle=np.array(data_angle)
            for key in indname:  
                if file_angle_name in key[4:10]:
                    position=key[13:]
                    position=int(position)
                    Max=len(data_angle)
                    data_angle_dict={}
                    if position-(lenth+1)<0 and position+lenth<Max:
                        data_angle_dict[key]=data_angle[0:(position+lenth)]
                    elif position+lenth>Max and position-(lenth+1)>0:
                        data_angle_dict[key]=data_angle[position-(lenth+1):Max]
                    elif position-(lenth+1)<0 and position+lenth>Max:
                        data_angle_dict[key]=data_angle[0:Max]
                    else:
                        data_angle_dict[key]=data_angle[position-(lenth+1):position+lenth]
                    data_phipsi_dict.update(data_angle_dict)
    
    seq_lenth=41
    data_angle_new=copy.deepcopy(data_phipsi_dict)            
    for name in list(data_angle_new.keys()):
        array_angle=data_angle_new[name]
        number_aa=array_angle.shape[0]
        if number_aa != seq_lenth:
            data_angle_new.pop(name)
        else:
            data_angle_new  
    data_angle_new = pd.DataFrame([data_angle_new]).T
    data_angle_new = data_angle_new.reset_index()
    data_angle_new.columns = ['name', 'angle']
    return data_angle_new

##############SASA##############
def get_ind_sasa(file_path_ind,file_path_sasa,lenth):
    '''
    file_path_sasa :A folder contains the predicted sasa information of each protein, 
    where each protein corresponds to a CSV file,named 'proteinname_dsspdata', 
    the last column is sasa
    '''  
    data_SASA_dict={}
    indname = pd.read_csv(file_path_ind)['name']
    for root, dirs, files in os.walk(file_path_sasa):
        for p in range(len(files)):
            q=file_path_sasa+'\\'+files[p]
            file_sasa_name = os.path.basename(q)
            file_sasa_name = file_sasa_name.split('_')[0]
            data_sasa=pd.read_csv(q,header=0,encoding="gbk")
            data_sasa=data_sasa.loc[:,['ACC']]
            data_sasa=np.array(data_sasa)
            for key in indname:               
                if file_sasa_name in key[4:10]:
                    position=key[13:]
                    position=int(position)
                    Max=len(data_sasa)
                    data_sasa_dict={}
                    if position-(lenth+1)<0 and position+lenth<Max:
                        data_sasa_dict[key]=data_sasa[0:(position+lenth)]
                    elif position+lenth>Max and position-(lenth+1)>0:
                        data_sasa_dict[key]=data_sasa[position-(lenth+1):Max]
                    elif position-(lenth+1)<0 and position+lenth>Max:
                        data_sasa_dict[key]=data_sasa[0:Max]
                    else:
                        data_sasa_dict[key]=data_sasa[position-(lenth+1):position+lenth]
                    data_SASA_dict.update(data_sasa_dict)
    
    seq_lenth=41
    data_sasa_new=copy.deepcopy(data_SASA_dict)            
    for name in list(data_sasa_new.keys()):
        #print(name)
        array_sasa=data_sasa_new[name]
        number_a=array_sasa.shape[0]
        if number_a != seq_lenth:
            data_sasa_new.pop(name)
        else:
            data_sasa_new  
    data_sasa_new = pd.DataFrame([data_sasa_new]).T
    data_sasa_new = data_sasa_new.reset_index()
    data_sasa_new.columns = ['name', 'sasa']
    return data_sasa_new

##############pssm##############
def get_ind_pssm(file_path_ind,file_path_pssm):
    '''
    file_path_sasa :A folder contains the pssm information of each protein, 
    where each protein corresponds to a txt file,named 'proteinname_residue_position'
    '''     
    data_PSSM_dict={}
    indname = pd.read_csv(file_path_ind)['name']
    for root, dirs, files in os.walk(file_path_pssm):
        for p in range(len(files)):
            q=file_path_pssm +'\\'+files[p]
            file_pssm_name = os.path.basename(q)
            file_pssm_name = file_pssm_name.split('.')[0]
            file_pssm_name = file_pssm_name.replace("_", "|")        
            pssm_array=np.zeros((41,20),dtype=float)
            f=open(q)
            data_pssm=f.readlines()

            del(data_pssm[41:])
            row=0
            for line in data_pssm:
                pssm_1=line.strip('\n').split(' ')
                pssm_1 = [i for i in pssm_1 if i]
                pssm_array[row:]=pssm_1[2:22]
                row+=1
                for key in indname:
                    data_pssm_dict={} 
                    if file_pssm_name in key:                   
                        data_pssm_dict[key]=pssm_array                
                        data_PSSM_dict.update(data_pssm_dict)
    data_PSSM_dict = pd.DataFrame([data_PSSM_dict]).T
    data_PSSM_dict = data_PSSM_dict.reset_index()
    data_PSSM_dict.columns = ['name', 'pssm']
    return data_PSSM_dict

