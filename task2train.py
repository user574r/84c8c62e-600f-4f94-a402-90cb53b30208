# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 03:14:18 2024

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

R=df[['6','7','target']]
R['Y2']=Y2
xgbm_model = XGBRegressor(**xgbm_params)
xgbm_model.fit(X[:20000],Y[:20000])


Y2=xgbm_model.predict(X)

def train(file,model_output_name='model.pkl'):
    train = pd.read_csv(file)#, usecols=features+[target_col])

    roles = {
        'target': target_col
    }

    task = Task('reg')
    automl = TabularUtilizedAutoML(task = task,
                                   timeout = TIMEOUT,
                                   cpu_limit = N_THREADS,
                                   reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS},
                                   general_params = {'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]}, # Use only linear models and LGBM and no CatBoost
                                   tuning_params = {'max_tuning_time': 5 * 60}, # Set 5 minutes for LGBM params tuning
                                   lgb_params = {'default_params': {'num_trees': 10000, 'learning_rate': 0.01, 'early_stopping_rounds': 100}, # Set smaller LR and more ES rounds
                                                 'freeze_defaults': True},
                                  )
    oof_pred = automl.fit_predict(train, roles = roles, verbose = 1)
    print('MSE score: {}'.format(mean_squared_error(train[target_col].values, oof_pred.data[:, 0], squared=False)))

    joblib.dump(xgbm_model, model_output_name)
    print("Model saved to " + model_output_name)


print('XGB')
evaluate(train[features].copy(), train[target_col].copy(), xgbm_model)

    
    
    
if __name__ == '__main__':
    mode = sys.argv[1]
    file_path = sys.argv[2]
    if mode == 'train':
        train(file_path)
    if mode == 'predict':
        predict(file_path)    


