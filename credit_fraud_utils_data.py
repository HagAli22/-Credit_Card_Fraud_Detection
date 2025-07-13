from typing import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
import os
import joblib
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_data(config):
    train=pd.read_csv(config['dataset']['train']['path'])
    val=pd.read_csv(config['dataset']['val']['path'])

    y_train=train[config['dataset']['target']]
    x_train=train.drop([config['dataset']['target']],axis=1)

    y_val=val[config['dataset']['target']]
    x_val=val.drop([config['dataset']['target']],axis=1)

    return x_train,y_train,x_val,y_val

def load_test(config):
    test=pd.read_csv(config['dataset']['test']['path'])
    y_test=test[config['dataset']['target']]
    x_test=test.drop([config['dataset']['target']],axis=1)
    return x_test,y_test

def scale_data(train,config,val=None):

    if isinstance(train,pd.DataFrame):
        train=train.values
    
    if val is not None:
        if isinstance(val,pd.DataFrame):
            val=val.values

    if config['preprocesing']['scaler_type'] =='robsut':
        scaler=RobustScaler()
    
    elif config['preprocesing']['scaler_type'] =='stander':
        scaler=StandardScaler()

    elif config['preprocesing']['scaler_type'] =='minmax':
        scaler=MinMaxScaler()
    else:
        raise ValueError("Invalid scaler type. Please choose 'minmax' or 'standard' or 'robust'.")
    
    train_scaed=scaler.fit_transform(train)
    
    
    val_scaled=None
    if val is not None:
        val_scaled=scaler.transform(val)
    print('sucess scale')
    return train_scaed,val_scaled

    
        
def do_balance(train,ytrain,config,k=2):

    if isinstance(train,pd.DataFrame):
        train=train.values

    if isinstance(ytrain,pd.DataFrame):
        ytrain=ytrain.values.ravel()
    
    print("Dataset before balancing:")
    print(f"Number of Non-fraud transactions: {len(ytrain[ytrain == 0])}")
    print(f"Number of fraud transactions:     {len(ytrain[ytrain == 1])}")

    if config['balanceing']['method']=='under':
        counter=Counter(ytrain)
        max_sz=counter[1]
        balance=RandomUnderSampler(sampling_strategy={0:max_sz*k},random_state=config['randomseed'])

    elif config['balanceing']['method']=='smote':
        counter=Counter(ytrain)
        min_sz=int(counter[0]/2)
        print("counter[1]",min_sz)
        balance=SMOTE(sampling_strategy={1:min_sz},k_neighbors=5,random_state=config['randomseed'])

    elif config['balanceing']['method']=='under_vs_over':
        counter=Counter(ytrain)
        min_sz=int(counter[0]//2)
        max_sz=int(counter[0]//2)

        over_sample=SMOTE(sampling_strategy={1:min_sz},k_neighbors=5,random_state=config['randomseed'])
        under_sample=RandomUnderSampler(sampling_strategy={0:max_sz},random_state=config['randomseed'])

        balance=Pipeline(steps=[('over',over_sample),('under',under_sample)])
    else:
        raise ValueError('please select type of Sampler from this : smote under_vs_over under')

    
    train,ytrain=balance.fit_resample(train,ytrain)
    

    print("\n Dataset after balancing:")
    print(f"Number of Non-fraud transactions: {len(ytrain[ytrain == 0])}")
    print(f"Number of fraud transactions:     {len(ytrain[ytrain == 1])}")

    return train,ytrain






# def under_over_sample(x,y):
#     counter=Counter(y)
#     min_sz=int(counter[1]//2)
#     max_sz=int(counter[1]//2)

#     over_sample=SMOTE(sampling_strategy={0:min_sz},k_neighbors=5,random_state=1)
#     under_sample=RandomUnderSampler(sampling_strategy={1:max_sz},random_state=1)

#     pip=Pipeline(steps=[('over',over_sample),('under',under_sample)])
#     x,y=pip.fit_resample(x,y)
#     return x,y

# def under_sample(x,y):
#     counter=Counter(y)
#     print(counter)
#     max_sz=counter[1]
#     factor=2
#     under_sample=RandomUnderSampler(sampling_strategy={0:max_sz*factor},random_state=1)
#     print('l')
#     x,y=under_sample.fit_resample(x,y)
#     print('k')
#     return x,y

# def over_sample(x,y):
#     counter=Counter(y)
#     min_sz=int(counter[1])
#     over_sample=SMOTE(sampling_strategy={0:min_sz},k_neighbors=5,random_state=1)
#     x,y=over_sample.fit_resample(x,y)
#     return x,y

# def preprocess_data(data,over=0,under=0,under_over=0):  
#     y=data['Class']
#     data.drop(['Class'],axis=1,inplace=True)
#     if over!=0:
#         data,y=over_sample(data,y)

#     elif under != 0:
#         data,y=under_sample(data,y)

#     elif under_over != 0:
#         print(y)
#         data,y=under_over_sample(data,y)
#     return data ,y








    







