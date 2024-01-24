# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 03:14:38 2024

@author: VMoiseienko
"""
import pandas as pd
import joblib
import sys



def predict(file, model_output_name='xgb_model.pkl', output='preds.csv'):
    data = pd.read_csv(file)#, usecols=features)
    automl = joblib.load(model_output_name)
    pred = automl.predict(data)
    data['pred'] = pred#.data[:, 0]
    data.to_csv(output, index=False)
    print('Predictions saved to '+output)

if __name__ == '__main__':
    file_path = sys.argv[1]
    try:
        model_name= sys.argv[2]
    except Exception :
        model_name='model.pkl'
    try:
        output_name= int(sys.argv[3])
    except Exception :
        output_name='Prediction.csv'
    predict(file_path,model_name,output_name)
    
    
# python predict.py hidden_test.csv mdl.pkl prediction.csv 