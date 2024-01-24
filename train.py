# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 03:41:50 2024

@author: VMoiseienko
"""


import pandas as pd
import joblib
import sys
import numpy as np

#from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
#from lightautoml.tasks import Task

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


N_THREADS = 4 # threads cnt for lightautoml
RANDOM_STATE = 1010 # fixed random state for various reasons
TIMEOUT = 60
N_FOLDS = 5


#features = #['6', '7', '26', '39', '35']

def train(file_path,model_output_name='xgb_model.pkl',N=20000):
    
    target_col = 'target'
    
    df=pd.read_csv('train.csv')
    
    Y=df[target_col]
    X=df.drop(target_col,axis=1)

    xgbm_params = {
        'n_estimators':100, 
        'max_depth':7, 
        'learning_rate':0.05, 
        'eval_metric': 'rmse',
        
        'random_state': 1010
    }
    
    xgbm_model = XGBRegressor(**xgbm_params)
    xgbm_model.fit(X[:N],Y[:N])
    joblib.dump(xgbm_model, model_output_name)
    print("Model saved to " + model_output_name)





if __name__ == '__main__':
    file_path = sys.argv[1]
    try:
        model_name= sys.argv[2]
    except Exception :
        model_name='model.pkl'
    try:
        N= int(sys.argv[3])
    except Exception :
        N=20000
    train(file_path,model_name,N)




#python train.py train.csv mdl 500 to run with 500 only rows to train and save model to mdl.pkl file

